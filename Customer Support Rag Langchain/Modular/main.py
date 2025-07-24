import os
from dotenv import load_dotenv
from typing import TypedDict

from loader import load_txt_as_documents
from splitter import split_text
from embeddings import get_embedding_model
from vectorstore import setup_qdrant
from retriever import retrieve_fn
from generator import generate
from graph_builder import build_graph

load_dotenv()

class GraphState(TypedDict):
    question: str
    context: str
    answer: str

raw_text = load_txt_as_documents("rag_service.txt")
documents = split_text(raw_text)

embedding_model = get_embedding_model()
db = setup_qdrant(embedding_model)
db.add_documents(documents)

retrieve = retrieve_fn(db)
app = build_graph(retrieve, generate, GraphState)

question = "I have an issue setting a different delivery address up"
result = app.invoke({"question": question})

print("Question:", result['question'])
print("Context:\n", result['context'][:500])
print("Answer:\n", result['answer'])
