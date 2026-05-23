from dotenv import load_dotenv
from pathlib import Path
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

pdf_path = Path(__file__).parent / "FullStack.pdf"

# Load this file in python program

loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

# Split the docs into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
  chunk_size = 500,
  chunk_overlap = 100
)

chunks = text_splitter.split_documents(documents=docs)

# Vector Embedding
embedding_model = NVIDIAEmbeddings(
    model="nvidia/nv-embed-v1",
)

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
      embedding=embedding_model,
      url=os.getenv("QDRANT_URL"),
      api_key=os.getenv("QDRANT_API_KEY"),
      collection_name="learning_rag",
      batch_size=5,
      timeout=120,
      force_recreate=True,
)

print("Indexing of documents done.....")