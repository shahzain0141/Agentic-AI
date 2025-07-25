from loader import load_txt_as_documents
from config import *
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from build_graph import build_graph

documents = load_txt_as_documents("rag_service.txt")

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
)
embedding_function = OpenAIEmbeddings()
db = Qdrant(client=client, collection_name=COLLECTION_NAME, embeddings=embedding_function)
db.add_documents(documents)

print("Uploaded documents to Qdrant.")

app = build_graph(db)
inputs = {
    "question": "How do I change the delivery address?",
    "recipient": "shahzain0066@gmail.com"
}
result = app.invoke(inputs)

print("Answer:", result["answer"])
