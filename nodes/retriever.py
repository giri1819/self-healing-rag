def retrieve(state: dict, retriever) -> dict:
    question = state["question"]

    docs = retriever.invoke(question)

    chunks = [
        doc.page_content
        for doc in docs
    ]
    print(chunks)
    print(f"📚 Retrieved {len(chunks)} chunks")
    
    return {
        **state,
        "retrieved_chunks": chunks
    }