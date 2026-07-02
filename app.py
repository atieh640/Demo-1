import streamlit as st
import anthropic
import json
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="HishoGames AI Support",
    page_icon="🎮",
    layout="centered"
)

# --- Fake Database ---
ORDERS = {
    "99321": {
        "customer": "Ahmad",
        "email": "ahmad@email.com",
        "product": "Spider-Man 2 PS5 Disc",
        "status": "delivered",
        "delivery_date": "2024-01-15",
        "condition": "defective - reported scratched"
    },
    "88432": {
        "customer": "Sara",
        "email": "sara@email.com",
        "product": "God of War Ragnarok PS5 Disc",
        "status": "in_transit",
        "delivery_date": "2024-01-20",
        "condition": "good"
    },
    "77541": {
        "customer": "Mike",
        "email": "mike@email.com",
        "product": "FIFA 25 PS5 Disc",
        "status": "processing",
        "delivery_date": "2024-01-22",
        "condition": "good"
    }
}

SUPPORT_TICKETS = []

# --- Database Functions ---
def get_order_status(order_id: str) -> dict:
    order = ORDERS.get(order_id)
    if order:
        return {"found": True, "order": order}
    return {"found": False, "message": f"Order {order_id} not found"}

def create_support_ticket(order_id: str, issue: str, priority: str) -> dict:
    ticket_id = f"TKT-{len(SUPPORT_TICKETS) + 1001}"
    ticket = {
        "ticket_id": ticket_id,
        "order_id": order_id,
        "issue": issue,
        "priority": priority,
        "status": "open"
    }
    SUPPORT_TICKETS.append(ticket)
    return {"success": True, "ticket_id": ticket_id}

def process_replacement(order_id: str, reason: str) -> dict:
    order = ORDERS.get(order_id)
    if not order:
        return {"success": False, "message": "Order not found"}
    return {
        "success": True,
        "message": f"Replacement approved for order {order_id}",
        "new_order_id": f"{order_id}-R",
        "estimated_delivery": "3-5 business days",
        "product": order["product"]
    }

def check_inventory(product_name: str) -> dict:
    inventory = {
        "spider-man 2": {"in_stock": True, "quantity": 45},
        "god of war ragnarok": {"in_stock": True, "quantity": 12},
        "fifa 25": {"in_stock": False, "quantity": 0},
        "call of duty modern warfare 3": {"in_stock": True, "quantity": 8},
        "hogwarts legacy": {"in_stock": True, "quantity": 23},
        "gran turismo 7": {"in_stock": True, "quantity": 31},
    }
    key = product_name.lower()
    result = inventory.get(key, {"in_stock": None, "quantity": None})
    if result["in_stock"] is None:
        return {"found": False, "message": f"Product '{product_name}' not found"}
    return {"found": True, "product": product_name, **result}

# --- Tools Definition ---
TOOLS = [
    {
        "name": "get_order_status",
        "description": "Look up the status of a customer order using the order ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order ID number"}
            },
            "required": ["order_id"]
        }
    },
    {
        "name": "create_support_ticket",
        "description": "Create a support ticket for a customer issue that needs follow-up.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order ID"},
                "issue": {"type": "string", "description": "Description of the issue"},
                "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]}
            },
            "required": ["order_id", "issue", "priority"]
        }
    },
    {
        "name": "process_replacement",
        "description": "Process a replacement for a defective or damaged product.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order ID"},
                "reason": {"type": "string", "description": "Reason for replacement"}
            },
            "required": ["order_id", "reason"]
        }
    },
    {
        "name": "check_inventory",
        "description": "Check if a PS5 game is in stock.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_name": {"type": "string", "description": "Name of the game"}
            },
            "required": ["product_name"]
        }
    }
]

# --- RAG Setup ---
@st.cache_resource
def setup_rag():
    """Initialize ChromaDB - cached so it only runs once"""
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    collection = chroma_client.get_or_create_collection(
        name="hishogames_knowledge",
        embedding_function=embedding_fn
    )

    # Load knowledge if empty
    if collection.count() == 0:
        documents = [
            "HishoGames return policy: Customers can return any game within 30 days if unopened. Opened games can only be returned if defective.",
            "HishoGames shipping: Standard shipping 3-5 business days. Express 1-2 days costs $9.99. Free shipping over $50.",
            "HishoGames PS5 games: Spider-Man 2, God of War Ragnarok, FIFA 25, Call of Duty Modern Warfare 3, Hogwarts Legacy, Gran Turismo 7.",
            "HishoGames prices: Spider-Man 2 $69.99, God of War Ragnarok $59.99, FIFA 25 $69.99, Call of Duty $69.99, Hogwarts Legacy $49.99, Gran Turismo 7 $39.99.",
            "Defective products: Scratched or non-working discs get free replacement. Customer must provide order number and photo evidence.",
            "Support hours: Monday to Friday 9am-6pm EST. Email support@hishogames.com or call 1-800-HISHO-PS5.",
            "Loyalty program: 1 point per dollar spent. 100 points = $5 discount. Points expire after 12 months.",
            "Payment methods: Visa, Mastercard, Amex, PayPal, HishoGames gift cards. No cryptocurrency.",
        ]
        collection.add(
            documents=documents,
            ids=[f"doc_{i}" for i in range(len(documents))]
        )

    return collection

# --- Core Agent Logic ---
def run_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "get_order_status":
        result = get_order_status(**tool_input)
    elif tool_name == "create_support_ticket":
        result = create_support_ticket(**tool_input)
    elif tool_name == "process_replacement":
        result = process_replacement(**tool_input)
    elif tool_name == "check_inventory":
        result = check_inventory(**tool_input)
    else:
        result = {"error": f"Unknown tool: {tool_name}"}
    return json.dumps(result)

def get_rag_context(query: str, collection) -> str:
    """Search knowledge base for relevant context"""
    results = collection.query(
        query_texts=[query],
        n_results=3,
        include=['documents', 'distances']
    )
    docs = results['documents'][0]
    distances = results['distances'][0]
    relevant = [
        doc for doc, dist
        in zip(docs, distances)
        if dist < 1.0
    ]
    return "\n".join(relevant) if relevant else ""

def chat(user_input: str, conversation_history: list, collection) -> tuple:
    """Main agent function combining RAG + Tools"""
    client = anthropic.Anthropic()

    # Get RAG context
    rag_context = get_rag_context(user_input, collection)

    # Build augmented message
    if rag_context:
        augmented = f"""Knowledge base context: {rag_context}

 Customer message: {user_input}"""
    else:
        augmented = user_input

    conversation_history.append({
        "role": "user",
        "content": augmented
    })

    system = """You are an intelligent customer support agent for HishoGames,
 a PS5 game store. You have access to company knowledge AND real order data.

 - Use your tools to look up real order information when needed
 - Answer policy questions from the context provided
 - Be proactive — if someone reports a defective product, process the replacement
 - Be friendly, concise, and resolve issues efficiently
 - Never make up order details or prices not in your context"""

    tools_used = []

    # Agent loop
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system,
            tools=TOOLS,
            messages=conversation_history
        )

        if response.stop_reason == "tool_use":
            conversation_history.append({
                "role": "assistant",
                "content": response.content
            })

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tools_used.append(block.name)
                    result = run_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            conversation_history.append({
                "role": "user",
                "content": tool_results
            })

        elif response.stop_reason == "end_turn":
            final_message = response.content[0].text
            conversation_history.append({
                "role": "assistant",
                "content": final_message
            })
            return final_message, tools_used, response.usage

# --- Streamlit UI ---

# Header
st.markdown("# 🎮 HishoGames")
st.markdown("### AI Customer Support")
st.markdown("---")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0

# Setup RAG
collection = setup_rag()

# Sidebar with info
with st.sidebar:
    st.markdown("### 🛠️ Agent Status")
    st.success("✅ AI Agent Online")
    st.info("📚 Knowledge Base Loaded")
    st.markdown("### 📋 Try asking:")
    st.markdown("- My disc is scratched, order #99321")
    st.markdown("- Is FIFA 25 in stock?")
    st.markdown("- What is your return policy?")
    st.markdown("- Check order #88432")
    st.markdown("---")
    st.markdown(f"**Tokens used:** {st.session_state.total_tokens}")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.session_state.total_tokens = 0
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"],
        avatar="🧑" if message["role"] == "user" else "🎮"):
        st.markdown(message["content"])
        if "tools_used" in message and message["tools_used"]:
            with st.expander("⚙️ Actions taken"):
                for tool in message["tools_used"]:
                    tool_labels = {
                        "get_order_status": "🔍 Looked up order",
                        "create_support_ticket": "🎫 Created support ticket",
                        "process_replacement": "🔄 Processed replacement",
                        "check_inventory": "📦 Checked inventory"
                    }
                    st.markdown(f"• {tool_labels.get(tool, tool)}")

# Chat input
if prompt := st.chat_input("How can we help you today?"):

    # Show user message
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Get agent response
    with st.chat_message("assistant", avatar="🎮"):
        with st.spinner("Looking into that for you..."):
            response, tools_used, usage = chat(
                prompt,
                st.session_state.conversation_history,
                collection
            )
        st.markdown(response)
        if tools_used:
            with st.expander("⚙️ Actions taken"):
                for tool in tools_used:
                    tool_labels = {
                        "get_order_status": "🔍 Looked up order",
                        "create_support_ticket": "🎫 Created support ticket",
                        "process_replacement": "🔄 Processed replacement",
                        "check_inventory": "📦 Checked inventory"
                    }
                    st.markdown(f"• {tool_labels.get(tool, tool)}")

    # Save response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "tools_used": tools_used
    })

    # Update token count
    st.session_state.total_tokens += usage.input_tokens + usage.output_tokens