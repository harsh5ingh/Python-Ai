
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))


from mem0 import Memory
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CLIENT = OpenAI(
  api_key=os.getenv("NVIDIA_API_KEY"),
  base_url="https://integrate.api.nvidia.com/v1"
)

config = {
  "version": "v1.1",
  "embedder": {
    "provider": "openai",
    "config": { "api_key": CLIENT, "model": "text-embedding-3-small"}
  },
  "llm": {
    "provider": "openai",
    "config": { "api_key": CLIENT, "model": "gpt-4.1"}
  },
  "vector_store": {
    "provider": "qdrant",
    "config": {
      "host": "localhost",
      "port": 6333
    }
  }
}

mem_client = Memory.from_config(config))