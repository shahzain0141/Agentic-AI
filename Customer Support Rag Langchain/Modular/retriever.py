def retrieve_fn(db):
    def retrieve(state):
        query = state["question"]
        retriever = db.as_retriever()
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"question": query, "context": context}
    return retrieve
