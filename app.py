from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from fastapi.responses import JSONResponse
import os

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from retriever import FAISSRetriever

# Get Ollama base URL from environment variable (defaults to localhost)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Initialize retriever
retriever = FAISSRetriever()

# Initialize LLM with configurable base URL
model = OllamaLLM(
    model="llama3.2",
    base_url=OLLAMA_BASE_URL
)

# Create the prompt template
template = """You are a helpful pedagogical assistant. Use the following context from the documents to answer the question at the end.

Context from documents:
{context}

Question: {question}

Provide a clear and helpful answer based on the context provided. If the context doesn't contain enough information to answer the question, say so."""
prompt = ChatPromptTemplate.from_template(template)

# Create the RAG chain
chain = (
    {"question": RunnablePassthrough()}
    | RunnablePassthrough.assign(context=lambda x: retriever.retrieve_documents(x["question"], k=5))
    | prompt
    | model
    | StrOutputParser()
)


app = FastAPI()

# Serve static files (CSS/JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve HTML
@app.get("/")
async def index():
    return FileResponse(os.path.join("static", "index.html"))




@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    question = data.get("question", "").strip()
    
    if not question:
        return JSONResponse({"answer": "Please provide a question."})

    try:
        # Call your RAG chain
        answer = chain.invoke({"question": question})
    except Exception as e:
        answer = "Sorry, something went wrong while processing your question."

    return JSONResponse({"answer": answer})

