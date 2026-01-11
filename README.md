# Conversational Chatbot

A RAG (Retrieval-Augmented Generation) based conversational chatbot built with FastAPI, LangChain, Ollama, and FAISS. This chatbot provides intelligent answers by retrieving relevant context from processed documents and generating responses using a local LLM.

## Project Structure

```
conversationalChatbot/
│
├── app.py                      # FastAPI web application with RAG chain and MLflow tracking
├── main.py                     # Terminal-based chatbot interface
├── process_data.py             # Data ingestion and preprocessing pipeline (PDF extraction, chunking, embeddings, FAISS index creation)
├── retriever.py                # FAISS-based document retriever for semantic search
├── mlflow_tracking.py          # MLflow tracking utilities for embeddings and LLM operations
│
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker image configuration for the application
├── docker-compose.yml          # Docker Compose configuration for multi-container setup
├── docker-compose.local.yml    # Local development Docker Compose configuration
├── .dockerignore              # Files to exclude from Docker builds
├── .gitignore                 # Git ignore patterns
│
├── data.dvc                   # DVC file for data versioning
│
├── static/                    # Frontend static files
│   ├── index.html            # Web UI HTML
│   ├── script.js             # Frontend JavaScript for chat interface
│   └── style.css             # CSS styling
│
└── data/                      # Data directory (created during processing)
    ├── raw/                   # Raw PDF documents (user-provided)
    └── processed/             # Processed data
        └── faiss_index/       # FAISS vector index and metadata
```

## Key Components

### Core Application Files

- **`app.py`**: FastAPI web server that serves the chatbot interface and handles `/ask` API endpoints. Integrates RAG chain with MLflow tracking for monitoring performance.

- **`main.py`**: Command-line interface for running the chatbot in terminal mode. Provides an interactive terminal-based chat experience.

- **`process_data.py`**: Data processing pipeline that:
  - Loads PDF documents from `data/raw/`
  - Splits documents into chunks using RecursiveCharacterTextSplitter
  - Generates embeddings using sentence-transformers
  - Creates and saves FAISS vector index for efficient similarity search
  - Tracks embedding operations with MLflow

- **`retriever.py`**: FAISSRetriever class that:
  - Loads the FAISS index and document chunks
  - Encodes user queries using the embedding model
  - Performs semantic search to retrieve relevant document chunks
  - Tracks retrieval operations with MLflow

- **`mlflow_tracking.py`**: MLflow tracking module with:
  - `EmbeddingTracker`: Tracks embedding creation and retrieval operations
  - `LLMTracker`: Tracks LLM inference and complete RAG chain operations
  - Experiment management utilities

### Frontend

- **`static/index.html`**: Web-based chat interface
- **`static/script.js`**: Client-side JavaScript for API communication and UI updates
- **`static/style.css`**: Styling for the chat interface

### Configuration & Deployment

- **`Dockerfile`**: Containerizes the FastAPI application
- **`docker-compose.yml`**: Orchestrates the application and Ollama service
- **`requirements.txt`**: Python package dependencies
- **`data.dvc`**: DVC configuration for data versioning

## Features

- **RAG Architecture**: Retrieval-Augmented Generation for context-aware responses
- **FAISS Vector Search**: Efficient semantic search using Facebook AI Similarity Search
- **MLflow Integration**: Comprehensive tracking of embeddings, retrieval, and LLM operations
- **Multiple Interfaces**: Web UI (FastAPI) and terminal interface
- **Docker Support**: Containerized deployment with Docker Compose
- **Data Versioning**: DVC integration for managing data versions
- **Local LLM**: Uses Ollama for running LLMs locally

## Technology Stack

- **Framework**: FastAPI
- **LLM**: Ollama (LangChain integration)
- **Vector Store**: FAISS
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Document Processing**: LangChain (PyPDFLoader, RecursiveCharacterTextSplitter)
- **Tracking**: MLflow
- **Data Versioning**: DVC
- **Containerization**: Docker, Docker Compose

## Start the app

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (for containerized deployment)
- Ollama installed and running (or use the Docker Compose setup)

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare your data:
   - Place PDF documents in `data/raw/`
   - Run the data processing pipeline:
     ```bash
     python process_data.py
     ```

4. Start the application:
   - **Web Interface**: `uvicorn app:app --reload`
   - **Terminal Interface**: `python main.py`
   - **Docker**: `docker-compose up`

## Configuration

### Environment Variables

- `OLLAMA_BASE_URL`: Ollama server URL (default: `http://localhost:11434`)
- `OLLAMA_MODEL`: LLM model name (default: `llama3.2`)
- `RETRIEVER_K`: Number of documents to retrieve (default: `5`)
- `MLFLOW_TRACKING_URI`: MLflow tracking URI (default: `file:./mlruns`)
- `MLFLOW_EXPERIMENT_NAME`: MLflow experiment name (default: `conversational-chatbot`)

## Usage

### Web Interface

1. Start the FastAPI server: `uvicorn app:app --reload`
2. Open your browser to `http://localhost:8000`
3. Type your question in the chat interface

### Terminal Interface

1. Run: `python main.py`
2. Type your questions in the terminal
3. Type `quit` or `exit` to stop

## Data Processing

The `process_data.py` script processes PDF documents:

- Extracts text from PDFs in `data/raw/`
- Splits documents into chunks (default: 200 characters with 50 character overlap)
- Generates embeddings using sentence-transformers
- Creates FAISS index for fast similarity search
- Saves processed data to `data/processed/faiss_index/`

## MLflow Tracking

The application tracks:
- Embedding creation metrics (duration, chunks per second)
- Retrieval operations (query encoding, search duration, similarity scores)
- LLM inference metrics (inference duration, tokens per second)
- Complete RAG chain metrics (total duration, retrieval vs LLM time)

View tracking data at the MLflow tracking URI (default: `./mlruns`).
