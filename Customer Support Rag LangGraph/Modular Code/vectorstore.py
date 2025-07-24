from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.vectorstores import Qdrant

def setup_qdrant(embedding_function):
    qdrant_client = QdrantClient(host="localhost", port=6333)
    qdrant_client.recreate_collection(
        collection_name="rag_txt_collection",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    db = Qdrant(client=qdrant_client, collection_name="rag_txt_collection", embeddings=embedding_function)
    return db
