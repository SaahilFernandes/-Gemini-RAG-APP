

# Gemini RAG Q&A Assistant

A multi-agent Q&A assistant powered by Google Gemini API that uses Retrieval-Augmented Generation (RAG) to answer questions from uploaded text documents.

## Architecture Overview

This application integrates several components to create an intelligent Q&A system:

1. **Data Ingestion Layer**
   - Accepts user-uploaded text files
   - Splits documents into manageable chunks
   - Maintains session state to track uploaded files

2. **Vector Database**
   - Uses FAISS for efficient vector similarity search
   - Stores document chunks with Gemini embeddings

3. **LLM Integration**
   - Leverages Google's Gemini 1.5 Flash for generating answers
   - Configurable temperature for response creativity

4. **Multi-Agent System**
   - Knowledge Base Agent: Answers questions about uploaded documents
   - Calculator Agent: Performs mathematical calculations
   - Dictionary Agent: Provides word definitions
   - Router Agent: Decides which specialized agent to use based on query type

5. **User Interface**
   - Built with Streamlit for intuitive interactions
   - Shows decision-making process and retrieved context for transparency

## Key Design Choices

1. **Vector Search Configuration**
   - Chunk size of 500 characters with 50-character overlap for context retention
   - Top-3 retrieval for each query to balance context and specificity

2. **Session State Management**
   - Persistent storage of uploaded files and retriever objects
   - Tracking of knowledge base build status

3. **Error Handling**
   - Comprehensive error handling for file uploads, processing, and API calls
   - Informative user feedback throughout the process

4. **Agent Framework**
   - Custom prompt template that guides the agent's decision-making process
   - Tool definitions with clear usage boundaries

5. **Temporary File Management**
   - Secure handling of uploaded files using Python's tempfile
   - Proper cleanup after processing

## Requirements

- Python 3.7+
- Google AI Studio API key

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install streamlit langchain langchain-google-genai langchain-community python-dotenv faiss-cpu
   ```
3. Create a `.env` file in the project root with your Google API key:
   ```
   GOOGLE_API_KEY=AI_studio_api_key
   ```

## Usage

1. Run the Streamlit application:
   ```
   streamlit run app.py
   ```
2. Upload text files (.txt) using the sidebar
3. Click "Build Knowledge Base from Uploaded Files"
4. Ask questions in the input field:
   - Regular questions will query the document knowledge base
   - Questions containing "calculate" will use the calculator
   - Questions containing "define" will use the dictionary

## Example Queries

- "What is Product A best known for?"
- "Calculate 157 * 42"
- "Define artificial intelligence"
