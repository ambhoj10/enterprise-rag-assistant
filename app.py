import streamlit as st
import os

from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Azure OpenAI LLM
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0
)

# Azure OpenAI Embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment="text-embedding-ada-002",
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# Streamlit Config
st.set_page_config(
    page_title="Enterprise Autonomous Multi-Agent AI Platform",
    page_icon="🧠",
    layout="wide"
)

# UI
st.title("🧠 Enterprise Autonomous Multi-Agent AI Platform")

st.caption(
    "Multi-document enterprise RAG with conversational memory, retrieval intelligence, and autonomous AI orchestration"
)

# Session Memory
if "chat_history" not in st.session_state:

    st.session_state.chat_history = []

# Persistent Vector DB
persist_directory = "chroma_db"

# Multi-file uploader
uploaded_files = st.file_uploader(
    "Upload Enterprise Documents",
    type="pdf",
    accept_multiple_files=True
)

# Process uploaded documents
if uploaded_files:

    all_docs = []

    for uploaded_file in uploaded_files:

        # Save uploaded file
        file_path = uploaded_file.name

        with open(file_path, "wb") as f:

            f.write(uploaded_file.read())

        # Load PDF
        loader = PyPDFLoader(file_path)

        documents = loader.load()

        all_docs.extend(documents)

    # Chunking Strategy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )

    docs = text_splitter.split_documents(all_docs)

    # Create Persistent Vector DB
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    st.success(
        f"✅ Processed {len(uploaded_files)} enterprise documents successfully!"
    )

# Load Persistent Vector Database
if os.path.exists(persist_directory):

    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    st.info(
        "📚 Enterprise knowledge base loaded successfully!"
    )

    # User Question
    question = st.text_input(
        "Ask a question across all enterprise documents"
    )

    if question:

        # Advanced Retriever
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 8,
                "fetch_k": 30,
                "lambda_mult": 0.7
            }
        )

        # Query Rewriting
        rewrite_prompt = f'''
You are an enterprise AI retrieval optimizer.

Rewrite the user's question into a more detailed and searchable enterprise query.

Focus on:
- Azure technologies
- enterprise terminology
- technical entities
- retrieval optimization

Original Question:
{question}

Optimized Search Query:
'''

        rewritten_query = llm.invoke(
            rewrite_prompt
        ).content

        # Display rewritten query
        st.subheader("🧠 Optimized Search Query")

        st.code(rewritten_query)

        # Retrieve Documents
        retrieved_docs = retriever.invoke(
            rewritten_query
        )

        # Conversation History
        history = "\n".join(
            st.session_state.chat_history
        )

        # Build Context
        context = "\n\n".join([
            doc.page_content
            for doc in retrieved_docs
        ])

        # Multi-Agent Workflow Trigger
        question_lower = question.lower()

        # Determine active agents
        active_agents = []

        # Security Agent Activation
        if any(keyword in question_lower for keyword in [
            "security",
            "rbac",
            "governance",
            "encryption",
            "acl",
            "compliance"
        ]):

            active_agents.append("Security Agent")

        # Analytics Agent Activation
        if any(keyword in question_lower for keyword in [
            "analytics",
            "pipeline",
            "synapse",
            "databricks",
            "spark",
            "data"
        ]):

            active_agents.append("Analytics Agent")

        # Monitoring Agent Activation
        if any(keyword in question_lower for keyword in [
            "monitor",
            "logging",
            "metrics",
            "observability",
            "performance"
        ]):

            active_agents.append("Monitoring Agent")

        # Default Agent
        if not active_agents:

            active_agents.append("General Enterprise Agent")

        # Display Active Agents
        st.subheader("🤖 Active AI Agents")

        for agent in active_agents:

            st.success(agent)

        # Store Agent Outputs
        agent_outputs = []

        # SECURITY AGENT
        if "Security Agent" in active_agents:

            security_prompt = f'''
You are a Security AI Agent.

Focus on:
- Azure RBAC
- governance
- encryption
- ACLs
- compliance
- enterprise security

Retrieved Context:
{context}

Question:
{question}

Provide:
- security analysis
- governance technologies
- compliance considerations
'''

            security_response = llm.invoke(
                security_prompt
            ).content

            agent_outputs.append(
                f"Security Agent Analysis:\n{security_response}"
            )

        # ANALYTICS AGENT
        if "Analytics Agent" in active_agents:

            analytics_prompt = f'''
You are an Analytics AI Agent.

Focus on:
- Azure Synapse Analytics
- Azure Databricks
- Spark
- pipelines
- ETL
- data engineering

Retrieved Context:
{context}

Question:
{question}

Provide:
- analytics architecture analysis
- data engineering technologies
- pipeline insights
'''

            analytics_response = llm.invoke(
                analytics_prompt
            ).content

            agent_outputs.append(
                f"Analytics Agent Analysis:\n{analytics_response}"
            )

        # MONITORING AGENT
        if "Monitoring Agent" in active_agents:

            monitoring_prompt = f'''
You are a Monitoring AI Agent.

Focus on:
- Azure Monitor
- logging
- observability
- telemetry
- metrics
- operational visibility

Retrieved Context:
{context}

Question:
{question}

Provide:
- monitoring analysis
- observability insights
- telemetry recommendations
'''

            monitoring_response = llm.invoke(
                monitoring_prompt
            ).content

            agent_outputs.append(
                f"Monitoring Agent Analysis:\n{monitoring_response}"
            )

        # GENERAL AGENT
        if "General Enterprise Agent" in active_agents:

            general_prompt = f'''
You are a General Enterprise AI Agent.

Retrieved Context:
{context}

Question:
{question}

Provide:
- enterprise knowledge synthesis
- technology overview
- professional summary
'''

            general_response = llm.invoke(
                general_prompt
            ).content

            agent_outputs.append(
                f"General Agent Analysis:\n{general_response}"
            )

        # Combine Agent Analyses
        combined_analysis = "\n\n".join(agent_outputs)

        # Coordinator Agent
        coordinator_prompt = f'''
You are a Coordinator AI Agent.

Your responsibilities:
- combine multiple AI agent analyses
- remove duplication
- synthesize enterprise insights
- generate one professional response

Conversation History:
{history}

Optimized Search Query:
{rewritten_query}

Agent Analyses:
{combined_analysis}

Current User Question:
{question}

Provide:
- unified enterprise response
- structured analysis
- professional summary
'''

        # Final Coordinated Response
        response = llm.invoke(
            coordinator_prompt
        )

        # Save Memory
        st.session_state.chat_history.append(
            f"User: {question}"
        )

        st.session_state.chat_history.append(
            f"AI: {response.content}"
        )

        # Display Final Answer
        st.subheader("📖 AI Answer")

        st.markdown(response.content)

        # Retrieval Diagnostics
        st.subheader("🔎 Retrieval Diagnostics")

        st.write(
            f"Retrieved {len(retrieved_docs)} chunks"
        )

        # Retrieved Context
        with st.expander("View Retrieved Context"):

            for i, doc in enumerate(retrieved_docs):

                st.markdown(f"### Chunk {i+1}")

                st.write(doc.page_content[:700])

        # Multi-Agent Analyses
        with st.expander("🤖 Multi-Agent Analyses"):

            for output in agent_outputs:

                st.markdown(output)

# Conversation Memory
with st.expander("🧠 Conversation Memory"):

    if st.session_state.chat_history:

        for item in st.session_state.chat_history:

            st.write(item)

    else:

        st.write("No conversation history yet.")