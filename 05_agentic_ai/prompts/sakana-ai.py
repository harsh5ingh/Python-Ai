import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

import os
from dotenv import load_dotenv
from openai import OpenAI

import json
load_dotenv()

client = OpenAI(
    api_key=os.getenv("FUGU_API_KEY"),
    base_url="https://api.sakana.ai/v1",
)

response = client.responses.create(
    model="fugu",
    input="Write a one-sentence haiku about Tokyo rain.",
)

print(response.output_text)