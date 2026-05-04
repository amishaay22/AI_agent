import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
from google import genai

# ---------------- ENV ----------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found in .env")
    st.stop()

client = genai.Client(api_key=GOOGLE_API_KEY)

# ---------------- UI ----------------
st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📄 Chat with your PDFs (Gemini)")

with st.sidebar:
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

# ---------------- SESSION ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# ---------------- FUNCTIONS ----------------
def extract_text_from_multiple_pdfs(files):
    text = ""
    for file in files:
        reader = PdfReader(file)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text

@st.cache_resource
def create_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_texts(chunks, embeddings)

# ---------------- PROCESS PDFs ----------------
if uploaded_files:
    with st.spinner("Processing PDFs..."):
        combined_text = extract_text_from_multiple_pdfs(uploaded_files)

        if not combined_text.strip():
            st.error("No readable text found.")
            st.stop()

        st.session_state.vector_store = create_vector_store(combined_text)

    st.success(f"✅ {len(uploaded_files)} PDFs processed successfully!")

# ---------------- CHAT ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask something...")

if user_input and st.session_state.vector_store:

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    docs = st.session_state.vector_store.similarity_search(user_input, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer ONLY using the context below.
If answer not found, say "Not found in document".

Context:
{context}

Question:
{user_input}
"""

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = client.models.generate_content(
                    model="gemini-flash-latest",
                    contents=prompt
                )
                answer = response.text
            except Exception as e:
                answer = f"❌ Error: {str(e)}"

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ---------------- WARNING ----------------
if user_input and not st.session_state.vector_store:
    st.warning("⚠️ Please upload at least one PDF first.")
    

# Please follow the below steps-

# 1. Create virtual environment: python -m venv venv
# 2. Activate Virtual Environment: .\venv\Scripts\activate
# 3. Install Required Libraries:
# pip install streamlit PyPDF2 langchain langchain-community python-dotenv faiss-cpu sentence-transformers langchain-together
# 4. streamlit run app.py