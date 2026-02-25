import os
import streamlit as st
from dotenv import load_dotenv
import tempfile

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ----------------------------
# Load API Key
# ----------------------------
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    st.error("Add OPENAI_API_KEY in .env file")
    st.stop()


# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📄 Dynamic PDF RAG Chatbot")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
question = st.text_input("Ask a question about the uploaded PDF:")


# ----------------------------
# Create Vector Store from Uploaded PDF
# ----------------------------
@st.cache_resource
def create_vectorstore(file_bytes):

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    return vectorstore


if uploaded_file is not None:

    vectorstore = create_vectorstore(uploaded_file.read())
    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_template(
        """
Answer the question only using the context below.

Context:
{context}

Question:
{question}
"""
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    if question:
        with st.spinner("Analyzing document..."):
            answer = rag_chain.invoke(question)

        st.write("### 📌 Answer:")
        st.write(answer)