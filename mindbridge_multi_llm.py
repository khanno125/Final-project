import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

#CONFIGURATION
DATA_PATH = r"E:\mindbridge"
INDEX_PATH = r"E:\mindbridge_faiss_index"

#API KEYS
# OpenAI
os.environ["OPENAI_API_KEY"] = ""
# Anthropic
os.environ["ANTHROPIC_API_KEY"] = ""
# Vertex / Gemini
os.environ["GEMINI_API_KEY"] = ""

#STREAMLIT
st.set_page_config("MindBridge", "🧠")
st.title("🧠 MindBridge")
st.caption("Smart Support for Emotional Well-Being")

#EMBEDDINGS/VECTORISATION
embeddings = OpenAIEmbeddings()
if not os.path.exists(INDEX_PATH):
    st.info("📚 Creating FAISS index from documents...")
    docs = []
    for f in os.listdir(DATA_PATH):
        if f.endswith(".txt"):
            with open(os.path.join(DATA_PATH, f), encoding="utf-8") as file:
                docs.append(Document(page_content=file.read()))

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(INDEX_PATH)
else:
    vectorstore = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

retriever = vectorstore.as_retriever(k=3)

#Dropdown LLM
llm_choice = st.selectbox(
    "Choose LLM",
    [
        "OpenAI GPT-3.5",
        "OpenAI GPT-4o-mini",
        "Claude 3.5 Sonnet",
        "Gemini/Vertex"
    ]
)

#LLM Selection
if llm_choice == "OpenAI GPT-3.5":
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

elif llm_choice == "OpenAI GPT-4o-mini":
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

elif llm_choice == "Claude 3.5 Sonnet":
    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.3)

else:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )

#Chatbot
query = st.text_input("Ask your question", value="")

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("Ask") and query:
    docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
You are MindBridge, an emotional support chatbot.
Do NOT diagnose or give medical advice.
Listen, validate the user's emotions, and provide supportive, empathetic guidance.

Context:
{context}


Question:
{query}
"""

    response = llm.invoke(prompt)
    st.session_state.history.append((query, response.content))

#DISPLAY 
for i, (q, a) in enumerate(reversed(st.session_state.history)):
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Assistant:** {a}")

    col1, col2, col3 = st.columns([1, 1, 6])
    with col1:
        st.button("👍", key=f"good_{i}")
    with col2:
        st.button("👎", key=f"bad_{i}")

    st.divider()
