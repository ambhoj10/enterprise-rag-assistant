import streamlit as st
import tempfile
import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma

from langchain_openai import AzureChatOpenAI

# Load environment variables
load_dotenv()

st.title("📚 Enterprise RAG Assistant")

st.caption(
    "AI-powered document question answering using Azure OpenAI, LangChain, and ChromaDB"
)

st.caption("Chat with enterprise documents using Azure OpenAI")

uploaded_file = st.file_uploader(
    "Upload Enterprise PDF Document",
    type=["pdf"]
)

question = st.text_input(
    "Ask questions from the uploaded document"
)

if uploaded_file:

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_pdf_path = temp_file.name

    # Load PDF
    loader = PyPDFLoader(temp_pdf_path)

    documents = loader.load()

    st.success(f"PDF loaded successfully! Total pages: {len(documents)}")

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    st.success(f"Total chunks created: {len(chunks)}")

    # Embeddings
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_deployment="text-embedding-ada-002",
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

    # Vector Store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    st.success("Embeddings generated and stored in ChromaDB!")

    # Retrieval + Answer Generation
    if question:

        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 20
            }
        )

        retrieved_docs = retriever.invoke(question)

        # Combine retrieved context
        context = "\n\n".join(
            [doc.page_content for doc in retrieved_docs]
        )

        # Azure OpenAI Chat Model
        llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            temperature=0
        )

        # Prompt
        prompt = f"""
You are an enterprise AI assistant.

Answer the user's question using the provided context.

If possible, summarize relevant information clearly and concisely.

If the answer is not available in the context, say:
"I could not find the answer in the uploaded document."

Context:
{context}

Question:
{question}
"""

        # Generate Answer
        with st.spinner("Generating AI answer..."):
            response = llm.invoke(prompt)

        # Display Answer
        st.subheader("📖 AI Answer")
        st.write(response)

        with st.expander("View Retrieved Context"):
            for i, doc in enumerate(retrieved_docs):
                st.markdown(f"### Chunk {i+1}")
                st.write(doc.page_content[:500])