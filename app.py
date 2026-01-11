from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from fastapi.responses import JSONResponse
import os
import time

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from retriever import FAISSRetriever
from mlflow_tracking import LLMTracker
import mlflow

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OllamaLLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")     


# Initialize retriever
retriever = FAISSRetriever()

# Initialize LLM
model = OllamaLLM(
    model=OllamaLLM_MODEL,
    base_url=OLLAMA_URL
)

# Create the prompt template
template = """You are a helpful pedagogical assistant. Use the following context from the documents to answer the question at the end.

Context from documents:
{context}

Question: {question}

Provide a clear and helpful answer based on the context provided. If the context doesn't contain enough information to answer the question, say so."""
prompt = ChatPromptTemplate.from_template(template)

# Wrapper function to track retrieval separately
def retrieve_with_tracking(question: str, k: int = 5):
    """Retrieve documents with timing for MLflow tracking."""
    retrieval_start = time.time()
    context = retriever.retrieve_documents(question, k=k)
    retrieval_duration = time.time() - retrieval_start
    return context, retrieval_duration


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
        # Track RAG chain execution with detailed timing
        total_start = time.time()
        
        # Retrieve context with tracking
        context, retrieval_duration = retrieve_with_tracking(question, k=5)
        
        # Format prompt
        formatted_prompt = prompt.format(context=context, question=question)
        
        # LLM inference with tracking
        llm_start = time.time()
        llm_response = model.invoke(formatted_prompt)
        llm_duration = time.time() - llm_start
        
        # Parse answer
        answer = StrOutputParser().parse(llm_response)
        
        total_duration = time.time() - total_start
        
        # Log to MLflow
        try:
            LLMTracker.log_rag_chain(
                model_name=OllamaLLM_MODEL,
                question=question,
                answer=answer,
                total_duration=total_duration,
                retrieval_duration=retrieval_duration,
                llm_duration=llm_duration,
                num_retrieved_docs=5,
                metadata={
                    "ollama_url": OLLAMA_URL,
                    "context_length": len(context),
                    "prompt_length": len(formatted_prompt),
                }
            )
            
            # Also log individual LLM inference
            LLMTracker.log_llm_inference(
                model_name=OllamaLLM_MODEL,
                prompt=formatted_prompt,
                response=answer,
                inference_duration=llm_duration,
                retrieval_duration=retrieval_duration,
                context_length=len(context),
                response_length=len(answer),
                metadata={
                    "ollama_url": OLLAMA_URL,
                }
            )
        except Exception as e:
            # Don't fail request if MLflow tracking fails
            print(f"Warning: MLflow tracking failed: {e}")
        
    except Exception as e:
        answer = f"Sorry, something went wrong while processing your question: {str(e)}"
        
        # Log error to MLflow
        try:
            with mlflow.start_run(run_name="rag_error", nested=True):
                mlflow.log_params({
                    "error_type": type(e).__name__,
                    "question": question[:500],  # Truncate if too long
                })
                mlflow.log_text(str(e), "error.txt")
        except:
            pass

    return JSONResponse({"answer": answer})

