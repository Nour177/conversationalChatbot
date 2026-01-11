"""
MLflow tracking utilities for the conversational chatbot.
Handles experiment tracking for embeddings and LLM operations.
"""

import os
import time
from typing import Dict, Any, Optional
import mlflow
from mlflow.tracking import MlflowClient


# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "conversational-chatbot")

# Initialize MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


class EmbeddingTracker:
    """MLflow tracker for embedding operations."""
    
    @staticmethod
    def log_embedding_creation(
        model_name: str,
        num_chunks: int,
        embedding_dim: int,
        chunk_size: int,
        chunk_overlap: int,
        duration: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log embedding creation metrics to MLflow.
        
        Args:
            model_name: Name of the embedding model
            num_chunks: Number of document chunks
            embedding_dim: Dimension of embeddings
            chunk_size: Size of chunks
            chunk_overlap: Overlap between chunks
            duration: Time taken to create embeddings
            metadata: Additional metadata to log
        """
        with mlflow.start_run(run_name="embedding_creation", nested=True):
            # Log parameters
            mlflow.log_params({
                "embedding_model": model_name,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "num_chunks": num_chunks,
                "embedding_dimension": embedding_dim,
            })
            
            # Log metrics
            mlflow.log_metrics({
                "embedding_creation_duration_seconds": duration,
                "chunks_per_second": num_chunks / duration if duration > 0 else 0,
            })
            
            # Log additional metadata
            if metadata:
                mlflow.log_params(metadata)
    
    @staticmethod
    def log_retrieval_operation(
        query: str,
        k: int,
        num_results: int,
        retrieval_duration: float,
        embedding_duration: float,
        search_duration: float,
        avg_similarity: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log retrieval operation metrics to MLflow.
        
        Args:
            query: The search query
            k: Number of documents requested
            num_results: Actual number of results returned
            retrieval_duration: Total retrieval time
            embedding_duration: Time to encode query
            search_duration: Time to search FAISS index
            avg_similarity: Average similarity score of results
            metadata: Additional metadata
        """
        with mlflow.start_run(run_name="retrieval_operation", nested=True):
            # Log parameters
            mlflow.log_params({
                "retrieval_k": k,
                "num_results": num_results,
                "query_length": len(query),
            })
            
            # Log metrics
            mlflow.log_metrics({
                "retrieval_duration_seconds": retrieval_duration,
                "embedding_duration_seconds": embedding_duration,
                "search_duration_seconds": search_duration,
            })
            
            if avg_similarity is not None:
                mlflow.log_metric("avg_similarity_score", avg_similarity)
            
            # Log query as artifact (truncated if too long)
            if len(query) < 1000:
                mlflow.log_text(query, "query.txt")
            
            # Log additional metadata
            if metadata:
                mlflow.log_params(metadata)


class LLMTracker:
    """MLflow tracker for LLM operations."""
    
    @staticmethod
    def log_llm_inference(
        model_name: str,
        prompt: str,
        response: str,
        inference_duration: float,
        retrieval_duration: Optional[float] = None,
        context_length: Optional[int] = None,
        response_length: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log LLM inference metrics to MLflow.
        
        Args:
            model_name: Name of the LLM model
            prompt: The prompt sent to the LLM
            response: The LLM response
            inference_duration: Time taken for inference
            retrieval_duration: Time taken for retrieval (if applicable)
            context_length: Length of context provided
            response_length: Length of response
            metadata: Additional metadata
        """
        with mlflow.start_run(run_name="llm_inference", nested=True):
            # Log parameters
            mlflow.log_params({
                "llm_model": model_name,
                "prompt_length": len(prompt),
                "context_length": context_length or len(prompt),
            })
            
            # Log metrics
            mlflow.log_metrics({
                "inference_duration_seconds": inference_duration,
                "tokens_per_second": (response_length or len(response)) / inference_duration if inference_duration > 0 else 0,
            })
            
            if retrieval_duration:
                mlflow.log_metric("retrieval_duration_seconds", retrieval_duration)
                mlflow.log_metric("total_rag_duration_seconds", inference_duration + retrieval_duration)
            
            if response_length:
                mlflow.log_metric("response_length", response_length)
            
            # Log prompt and response as artifacts (truncated if too long)
            if len(prompt) < 5000:
                mlflow.log_text(prompt, "prompt.txt")
            if len(response) < 5000:
                mlflow.log_text(response, "response.txt")
            
            # Log additional metadata
            if metadata:
                mlflow.log_params(metadata)
    
    @staticmethod
    def log_rag_chain(
        model_name: str,
        question: str,
        answer: str,
        total_duration: float,
        retrieval_duration: float,
        llm_duration: float,
        num_retrieved_docs: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log complete RAG chain operation to MLflow.
        
        Args:
            model_name: Name of the LLM model
            question: User's question
            answer: Final answer
            total_duration: Total time for RAG chain
            retrieval_duration: Time for retrieval
            llm_duration: Time for LLM inference
            num_retrieved_docs: Number of documents retrieved
            metadata: Additional metadata
        """
        with mlflow.start_run(run_name="rag_chain"):
            # Log parameters
            mlflow.log_params({
                "llm_model": model_name,
                "question_length": len(question),
                "answer_length": len(answer),
                "num_retrieved_docs": num_retrieved_docs,
            })
            
            # Log metrics
            mlflow.log_metrics({
                "total_duration_seconds": total_duration,
                "retrieval_duration_seconds": retrieval_duration,
                "llm_duration_seconds": llm_duration,
                "retrieval_ratio": retrieval_duration / total_duration if total_duration > 0 else 0,
            })
            
            # Log question and answer as artifacts
            if len(question) < 1000:
                mlflow.log_text(question, "question.txt")
            if len(answer) < 5000:
                mlflow.log_text(answer, "answer.txt")
            
            # Log additional metadata
            if metadata:
                mlflow.log_params(metadata)


def get_or_create_experiment(experiment_name: str) -> str:
    """
    Get or create an MLflow experiment.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Experiment ID
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            return experiment_id
        return experiment.experiment_id
    except Exception as e:
        print(f"Warning: Could not create/get experiment: {e}")
        return "0"
