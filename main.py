from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from retriever import FAISSRetriever

# Initialize retriever
retriever = FAISSRetriever()

# Initialize LLM
model = OllamaLLM(model="llama3.2")

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

# Terminal interface
def main():
    """Main function to run the chatbot in terminal."""
    print("=" * 60)
    print("DevOps Training Chatbot")
    print("=" * 60)
    print("Type 'quit' or 'exit' to stop the chatbot.\n")
    
    while True:
        question = input("You: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        print("\nBot: ", end="", flush=True)
        try:
            response = chain.invoke({"question": question})
            print(response)
        except Exception as e:
            print(f"Error: {e}")
        
        print()  # Empty line for readability

if __name__ == "__main__":
    main()