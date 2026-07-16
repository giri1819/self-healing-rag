import os
from dotenv import load_dotenv
from vectorstore import build_vectorstore
from graph import build_graph

load_dotenv()

def run_pipeline(question: str, doc_path: str):
    print(f"\n{'='*50}")
    print(f"❓ Question: {question}")
    print('='*50)
    
    # Build components
    retriever = build_vectorstore(doc_path)
    pipeline = build_graph(retriever)
    
    # Initial state
    initial_state = {
        "question": question,
        "retrieved_chunks": [],
        "answer": "",
        "is_grounded": False,
        "retry_count": 0,
        "final_answer": None
    }
    
    # Run the self-healing pipeline
    result = pipeline.invoke(initial_state)
    
    print(f"\n{'='*50}")
    print(f"✅ FINAL ANSWER:\n{result['final_answer']}")
    print('='*50)

if __name__ == "__main__":
    run_pipeline(
        question="What are the refund rules for Moreathons?",
        doc_path="your_document.txt"
    )