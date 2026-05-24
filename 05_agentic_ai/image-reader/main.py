from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

response = client.chat.completions.create(
    model="meta/llama-3.2-90b-vision-instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Generate a caption for this image in about 50 words"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://images.pexels.com/photos/9553909/pexels-photo-9553909.jpeg"
                    }
                }
            ]
        }
    ],
    max_tokens=200
)

print(response.choices[0].message.content)