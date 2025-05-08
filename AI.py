import os
import streamlit as st
from dotenv import load_dotenv
import tempfile  # To handle uploaded files safely

# Langchain components
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain_experimental.utilities.python import PythonREPL
from langchain_core.messages import HumanMessage

# ---- 1. Load API Key ----
load_dotenv()
# Make sure your GOOGLE_API_KEY is set in your .env file

# Initialize session state variables
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []
if "kb_built" not in st.session_state:
    st.session_state.kb_built = False


# ---- 2. Data Ingestion and Vector Store Setup ----
def setup_vector_store(uploaded_files):
    """
    Loads documents from uploaded files, chunks them, creates embeddings,
    and builds a FAISS vector store. Returns a retriever object.
    """
    all_docs = []
    temp_file_paths = []

    if not uploaded_files:
        st.warning("Please upload at least one text file.")
        return None

    for uploaded_file in uploaded_files:
        # Save file content to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="wb") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_paths.append(tmp_file.name)
            print(f"Uploaded file '{uploaded_file.name}' saved to temp file: {tmp_file.name}")

        # Load file content
        try:
            loader = TextLoader(tmp_file.name, encoding='utf-8')
            loaded_docs = loader.load()
            if loaded_docs:
                all_docs.extend(loaded_docs)
                print(f"Successfully loaded {len(loaded_docs)} documents from {uploaded_file.name}")
            else:
                print(f"No documents loaded from {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")
            print(f"Error loading {uploaded_file.name}: {e}")

    if not all_docs:
        st.error("Could not load any documents from the uploaded files.")
        for path in temp_file_paths:
            try:
                os.unlink(path)
            except Exception as e:
                print(f"Error deleting temp file {path}: {e}")
        return None

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(all_docs)
    print(f"Created {len(chunks)} chunks from {len(all_docs)} documents")

    if not chunks:
        st.error("No text chunks could be created from the documents. Are the files empty or unreadable?")
        for path in temp_file_paths:
            try:
                os.unlink(path)
            except Exception as e:
                print(f"Error deleting temp file {path}: {e}")
        return None

    # Create embeddings and vector store
    try:
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        print("Creating new FAISS index from uploaded files...")
        vector_store = FAISS.from_documents(chunks, embeddings_model)
        print(f"FAISS index created successfully from {len(uploaded_files)} file(s).")
    except Exception as e:
        st.error(f"Error creating FAISS vector store: {e}")
        for path in temp_file_paths:
            try:
                os.unlink(path)
            except Exception as e_del:
                print(f"Error deleting temp file {path}: {e_del}")
        return None

    # Clean up temporary files
    for path in temp_file_paths:
        try:
            os.unlink(path)
            print(f"Temporary file {path} deleted.")
        except Exception as e:
            print(f"Error deleting temp file {path}: {e}")

    # Return retriever
    return vector_store.as_retriever(search_kwargs={"k": 3})


# ---- 3. LLM Integration ----
def get_llm():
    """Initialize the Gemini LLM"""
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.1,
        convert_system_message_to_human=True,
    )


# ---- 4. Tool Functions ----
def run_rag_chain(query: str):
    """Run the RAG chain for document Q&A"""
    # Make sure we have a retriever
    if not st.session_state.retriever:
        return {
            "result": "Knowledge base not initialized. Please upload documents and build the knowledge base first using the sidebar.",
            "source_documents": []}

    # Set up the QA chain
    llm = get_llm()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=st.session_state.retriever,
        return_source_documents=True
    )

    # Run the chain
    print(f"Running RAG chain for query: {query}")
    result = qa_chain.invoke({"query": query})
    return result


def calculate_tool_function(expression_string: str):
    """Calculator tool function"""
    python_repl = PythonREPL()
    try:
        result = python_repl.run(f"print({expression_string})")
        print(f"Calculator result for '{expression_string}': {result}")
        return result
    except Exception as e:
        error_msg = f"Error during calculation: {str(e)}. Ensure the input is a direct Python math expression."
        print(error_msg)
        return error_msg


def define_word_tool_function(word: str):
    """Dictionary tool function"""
    prompt_text = f"Please provide a concise definition for the word: {word}"
    llm = get_llm()
    response = llm.invoke([HumanMessage(content=prompt_text)])
    print(f"Definition for '{word}': {response.content}")
    return response.content


# ---- Main Application Logic ----
def main():
    # Page configuration
    st.set_page_config(page_title="Gemini RAG Q&A Assistant", layout="wide")
    st.title("🚀 Gemini-Powered Multi-Agent Q&A Assistant")
    st.write("""
        Upload your text documents (.txt files) to build a knowledge base.
        Then, ask questions about the content, perform calculations, or get word definitions!
    """)

    # Sidebar for document upload and knowledge base creation
    with st.sidebar:
        st.header("📚 Build Knowledge Base")

        # File uploader
        uploaded_files = st.file_uploader(
            "Upload your text files (.txt)",
            type=["txt"],
            accept_multiple_files=True
        )

        # Store uploaded files in session state
        if uploaded_files:
            file_names = [file.name for file in uploaded_files]
            if file_names != st.session_state.uploaded_files_list:
                st.session_state.uploaded_files_list = file_names
                st.session_state.kb_built = False

        # Button to build knowledge base
        if st.button("Build Knowledge Base from Uploaded Files", key="build_kb"):
            if uploaded_files:
                with st.spinner("Processing documents and building knowledge base..."):
                    retriever_obj = setup_vector_store(uploaded_files)
                    if retriever_obj:
                        st.session_state.retriever = retriever_obj
                        st.session_state.kb_built = True
                        st.success("Knowledge base built successfully!")
                    else:
                        st.error("Failed to build knowledge base. Check logs or file contents.")
                        st.session_state.kb_built = False
            else:
                st.warning("Please upload at least one .txt file.")
                st.session_state.kb_built = False

    # Check if Knowledge Base is ready
    kb_ready = st.session_state.kb_built and st.session_state.retriever is not None

    # Status indicator
    if kb_ready:
        st.success("✅ Knowledge Base is ready. Ask your questions below!")
    else:
        st.info("Please upload documents and click 'Build Knowledge Base' in the sidebar to begin.")

    # Define the tools
    tools = [
        Tool(
            name="KnowledgeBaseQA",
            func=run_rag_chain,
            description="Use this tool when you need to answer questions about the content of the uploaded documents (e.g., products, company information, FAQs). Input should be the user's full question.",
        ),
        Tool(
            name="Calculator",
            func=calculate_tool_function,
            description="Use this tool ONLY when you need to perform mathematical calculations. Input MUST be a Python mathematical expression string (e.g., '2+2', '10*5/2'). Do NOT use for word problems or questions containing text other than the expression.",
        ),
        Tool(
            name="Dictionary",
            func=define_word_tool_function,
            description="Use this tool when you need to find the definition of a specific word. Input should be the single word to define (e.g., 'serendipity').",
        )
    ]

    # Agent prompt template
    agent_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", (
                "You are a helpful assistant. You have access to the following tools: KnowledgeBaseQA, Calculator, Dictionary.\n"
                "Based on the user's input, decide if a tool is appropriate. If so, respond with the tool call. If not, respond to the user directly.\n"
                " - For questions about products, company info, or FAQs from documents, use 'KnowledgeBaseQA'.\n"
                " - For math calculations (e.g., '5+5', 'calculate 10*3'), use 'Calculator' with ONLY the math expression as input.\n"
                " - For word definitions (e.g., 'define apple'), use 'Dictionary' with ONLY the word as input.\n"
                "When using KnowledgeBaseQA, the tool will return a dictionary containing the 'result' (the answer) and 'source_documents'. Present the 'result' as the main answer. You can mention that the information came from the knowledge base."
                "If the KnowledgeBaseQA tool says 'Knowledge base not initialized', inform the user they need to upload documents and build the knowledge base first using the sidebar."
                "If a tool is used, provide the answer from the tool. Do not just state which tool you used, but give the actual answer."
                "If no tool is suitable, answer as best you can or say you cannot answer."
            )),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Create the agent
    llm = get_llm()
    agent = create_tool_calling_agent(llm, tools, agent_prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    # User input
    query = st.text_input(
        "Ask your question:",
        placeholder="e.g., What are features of Product A? or Calculate 5*8",
    )

    # Process the query
    if query:
        st.markdown("---")
        st.write("🤖 Gemini is thinking...")

        try:
            # Special case for knowledge base questions when KB is not ready
            if not kb_ready and not any(
                    keyword in query.lower() for keyword in ["calculate", "define", "what does", "mean"]):
                st.warning(
                    "Please build the knowledge base using the sidebar before asking questions about document content.")
            else:
                # Execute the agent
                response = agent_executor.invoke({"input": query})
                final_answer = response.get("output",
                                            "Sorry, Gemini couldn't formulate a direct answer. Check console logs.")
                intermediate_steps = response.get("intermediate_steps", [])

                # Display the final answer
                st.subheader("✨ Gemini's Final Answer:")
                st.markdown(final_answer)

                # Display decision log and context
                if intermediate_steps:
                    st.markdown("---")
                    st.subheader("⚙️ Agent Decision Log & Context:")
                    for step in intermediate_steps:
                        action = step[0]
                        observation = step[1]

                        # Handle tool calls
                        if hasattr(action, 'tool_calls') and action.tool_calls:
                            for tool_call in action.tool_calls:
                                st.write(f"**Decision:** Using tool `{tool_call.name}` with input `{tool_call.args}`")

                                # For KnowledgeBaseQA, show retrieved documents
                                if tool_call.name == "KnowledgeBaseQA":
                                    if isinstance(observation, dict):
                                        retrieved_docs = observation.get("source_documents", [])
                                        if retrieved_docs:
                                            st.write("**Retrieved Context Snippets:**")
                                            for i, doc_obj in enumerate(retrieved_docs):
                                                st.markdown(f"**Snippet {i + 1} (from an uploaded file):**")
                                                st.caption(doc_obj.page_content)
                                    else:
                                        st.write(f"Tool Observation (KnowledgeBaseQA): {observation}")
                                else:
                                    st.write(f"**Tool Output ({tool_call.name}):** {observation}")

                        # Handle legacy format
                        elif hasattr(action, 'tool') and hasattr(action, 'tool_input'):
                            st.write(
                                f"**Decision (Legacy):** Using tool `{action.tool}` with input `{action.tool_input}`")
                            st.write(f"**Tool Output ({action.tool}):** {observation}")

                # No tools were used case
                elif not intermediate_steps and final_answer and "Sorry" not in final_answer and "I can't" not in final_answer:
                    st.write("")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("Please ensure your GOOGLE_API_KEY is correctly set and valid.")
            import traceback
            st.text(traceback.format_exc())


if __name__ == "__main__":
    main()