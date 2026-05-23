from dotenv import load_dotenv
import os
from langchain_nvidia import NVIDIAEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# Vector Embeddings
embedding_model = NVIDIAEmbeddings(
    model="nvidia/nv-embed-v1",
)

vector_db = QdrantVectorStore.from_existing_collection(
  url=os.getenv("QDRANT_URL"),
  api_key=os.getenv("QDRANT_API_KEY"),
  collection_name="learning_rag",
  embedding=embedding_model,
)


# Take user input
user_query = input("Ask Something: ")

# Relevant chunks from the vector db

search_result = vector_db.similarity_search(query=user_query)

context = "\n\n\n".join([f"Page Content: {result.page_content}\n Page Number: {result.metadata['page_label']}\n File Location: {result.metadata['source']}" for result in search_result ])

SYSTEM_PROMPT = """
You are a helpfull AI Assistant who answers user query based on the available context retrieved from a PDF file along with page_contents and page number.

You should only answer the user based on the following context and navigate the user to open the right page number to know more.

"""

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"""
            Context:
            {context}

            Question:
            {user_query}
            """
        },
    ]
)

print(f"🤖: {response.choices[0].message.content}")