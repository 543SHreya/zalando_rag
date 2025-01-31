import streamlit as st
import openai
import json
import os

# ---------------------------
# 1) Load environment variables if needed:
# from dotenv import load_dotenv
# load_dotenv()
# ---------------------------

# Set your OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY", "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")

# ---------------------------
# 2) Load Preprocessed Data
# ---------------------------
PREPROCESSED_DATA_PATH = "preprocessed_data.json"
try:
    with open(PREPROCESSED_DATA_PATH, "r", encoding="utf-8") as f:
        preprocessed_data = json.load(f)
except FileNotFoundError:
    preprocessed_data = []

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(
    page_title="Zalando Financial RAG Assistant",
    page_icon="🛍️",
    layout="wide",
)

# ---------------------------
# 3) Inject Custom CSS for Orange-Based Theme
# ---------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #ffffff;
    }

    .stButton>button {
        background-color: #ff6900 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        border: none !important;
    }

    .stTextInput>div>div>input {
        border: 2px solid #ff6900 !important;
    }

    h1 {
        color: #333333 !important; /* Darker text color for better visibility */
        font-size: 36px !important;
        font-weight: bold;
    }

    .stMarkdown p {
        color: #000000 !important;
        font-size: 16px !important;
    }

    .answer-box {
        border: 2px solid #ff6900;
        padding: 10px;
        border-radius: 5px;
        color: #000000;
        font-size: 16px;
        background-color: #f9f9f9;
    }

    .side-img {
        margin-top: 20px;
        padding-left: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# 4) Display Logo and Title
# ---------------------------
if os.path.exists("zalando_logo.png"):
    st.image("zalando_logo.png", width=150)

st.title("Zalando Financial RAG Assistant")
st.write("""
Ask questions about Zalando’s financial data. The system will consider **all** chunks of text before answering.
""")

# ---------------------------
# 5) Utility: Combine All Chunks
# ---------------------------
def get_combined_context() -> str:
    if not preprocessed_data:
        return "No data found."
    all_texts = [record["text"] for record in preprocessed_data]
    return "\n\n".join(all_texts)

# ---------------------------
# 6) RAG Pipeline
# ---------------------------
def answer_query_with_all_chunks(query: str, context="") -> str:
    combined_context = context if context else get_combined_context()

    system_prompt = """
You are a knowledgeable financial analyst with deep expertise in interpreting corporate financial reports, particularly Zalando’s. You must rely ONLY on the provided context for your answers and avoid including any information not explicitly found in that context.
"""

    user_prompt = f"Context:\n{combined_context}\n\nQuestion: {query}"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=1000
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"Error calling OpenAI API: {str(e)}"

# ---------------------------
# 7) Dynamic Agent-Based Questions
# ---------------------------
AGENTS = {
    "Financial Analyst": {
        "role_description": "You are a financial analyst at Zalando. Your role is to ask questions about the financial metrics of Zalando for Q3 2024. Frame your questions in a conversational format, focusing on metrics such as revenue, margins, and regional performance.",
    },
    "Marketing Specialist": {
        "role_description": "You are a marketing specialist at Zalando. Your role is to ask questions about customer acquisition, conversion rates, and campaign effectiveness for Q3 2024. Frame your questions in a conversational and insightful manner.",
    },
    "Strategy Manager": {
        "role_description": "You are a strategy manager at Zalando. Your role is to ask questions about growth drivers, market share, and strategic investments for Q3 2024. Frame your questions with a focus on long-term business strategy.",
    }
}

def generate_dynamic_questions(agent_role: str):
    agent_prompt = AGENTS[agent_role]["role_description"] + " Generate four questions dynamically in a conversational manner."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that helps generate questions."},
                {"role": "user", "content": agent_prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        questions = response.choices[0].message["content"].split("\n")
        return [q.strip() for q in questions if q.strip()]
    except Exception as e:
        return [f"Error generating questions: {str(e)}"]

def simulate_persona_conversation(agent_role: str):
    questions = generate_dynamic_questions(agent_role)
    conversation = []

    for question in questions:
        response = answer_query_with_all_chunks(question)
        conversation.append((question, response))

    return conversation

# ---------------------------
# Main UI
# ---------------------------
if not preprocessed_data:
    st.error("No preprocessed data found. Please ensure 'preprocessed_data.json' is present.")
else:
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Ask a Custom Question")
        custom_query = st.text_input("Enter your question:", "")
        if st.button("Ask"):
            if not custom_query.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Generating answer..."):
                    answer = answer_query_with_all_chunks(custom_query)
                st.markdown(f"<div class='answer-box'><strong>Answer:</strong><br>{answer}</div>", unsafe_allow_html=True)

        st.write("---")

        st.subheader("Simulated Persona Interaction")
        selected_agent = st.selectbox("Choose an agent for simulation:", options=list(AGENTS.keys()))
        if st.button("Simulate Conversation"):
            with st.spinner("Simulating conversation..."):
                conversation = simulate_persona_conversation(selected_agent)
                for question, response in conversation:
                    st.markdown(f"**Question:** {question}")
                    st.markdown(f"**Response:** {response}")
                    st.write("---")

    with col2:
        if os.path.exists("zalando_side.png"):
            st.image("zalando_side.png", width=300)
        else:
            st.info("Side image not found. Please add zalando_side.png.")

    st.write("---")
    st.subheader("Disclaimer")
    st.write("""
    All data is drawn from internal or publicly available Zalando reports. For official decisions, always consult the original statements. This tool provides AI-generated information for reference purposes only.
    """)

    st.write("**Zalando RAG Demo** | Powered by [Streamlit](https://streamlit.io/) & [OpenAI](https://openai.com/).")
