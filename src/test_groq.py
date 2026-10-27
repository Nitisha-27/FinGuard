
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in the project root .env file.")

client = Groq(api_key=api_key)

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "system",
            "content": (
                "You classify financial statements into exactly one category: "
                "Informational, Advisory, or Uncertain. "
                "Informational means explaining or describing financial concepts. "
                "Advisory means recommending or directing a financial action. "
                "Uncertain means the text is too ambiguous or lacks sufficient "
                "context to determine its communicative intent. "
                "Return only the category name."
            ),
        },
        {
            "role": "user",
            "content": (
                "Classify this statement: "
                "'A diversified portfolio can reduce exposure to individual company risk.'"
            ),
        },
    ],
    temperature=0,
)

print("Groq connection successful!")
print("Model: openai/gpt-oss-20b")
print("Classification:", response.choices[0].message.content.strip())

