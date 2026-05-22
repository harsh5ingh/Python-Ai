from dotenv import load_dotenv
from langchain_nvidia import NVIDIAEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

# Vector Embeddings
embedding_model = NVIDIAEmbeddings(
    model="nvidia/nv-embed-v1",
)

vector_db = QdrantVectorStore.from_existing_collection(
  url="http://localhost:6333",
  collection_name="learning-rag",
  embedding=embedding_model,
)

