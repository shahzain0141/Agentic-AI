from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from typing import TypedDict
from langchain_community.vectorstores import Qdrant
from email_tool import send_email
from config import COLLECTION_NAME
from langchain_core.documents import Document

class GraphState(TypedDict):
    question: str
    context: str
    answer: str
    recipient: str

llm = ChatOpenAI(model="gpt-3.5-turbo")

def get_retrieve_node(db: Qdrant):
    def retrieve(state: GraphState):
        query = state["question"]
        retriever = db.as_retriever()
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"question": query, "context": context}
    return RunnableLambda(retrieve)

def generate(state: GraphState):
    prompt = f"""Answer the question using this context:\n\n{state['context']}\n\nQuestion: {state['question']}"""
    response = llm.invoke(prompt)
    return {
        "question": state["question"],
        "context": state["context"],
        "answer": response.content,
        "recipient": state["recipient"]
    }

def send_email_node(state: GraphState):
    result = send_email.invoke({
        "recipient": state["recipient"],
        "question": state["question"],
        "answer": state["answer"]
    })
    print(result)
    return state
