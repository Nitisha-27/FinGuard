
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from groq import Groq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

MODEL = "openai/gpt-oss-120b"
LABELS = {"Informational", "Advisory", "Uncertain"}

app = FastAPI(
    title="FinGuard API",
    description="Financial response classification service",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://fin-guard-ten.vercel.app",
        "https://finguard-ten.vercel.app",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClassificationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    context: str = Field(
        default="Standalone financial statement; surrounding source context is not provided.",
        max_length=5000,
    )


class ClassificationResponse(BaseModel):
    label: str
    explanation: str
    model: str


def build_prompt():
    return (
        "Classify the supplied financial text into exactly one label.\n\n"

        "Informational: The text primarily reports or explains a financial "
        "fact, result, trend, concept, comparison, product feature, risk, "
        "or assessment. It does not recommend a financial action. "
        "A statement can mention risk or uncertainty and still be Informational "
        "if its main purpose is to describe or report information.\n\n"

        "Advisory: The text explicitly or implicitly recommends, encourages, "
        "discourages, or favors a financial action, decision, or choice. "
        "Look for advice, suggested actions, or language indicating what "
        "someone should, could, or ought to do.\n\n"

        "Uncertain: Use when the text expresses an unresolved prediction, "
        "indecision, or inability to determine a financial outcome, and "
        "its primary purpose is to communicate that uncertainty rather "
        "than report an established fact or recommend an action. Examples "
        "include uncertainty about whether a stock will rise or fall, "
        "whether growth will continue, or which outcome will occur. "
        "Do not automatically classify every sentence containing words "
        "such as 'may', 'might', or 'uncertain' as Uncertain; classify "
        "according to the text's main communicative purpose.\n\n"

        "Decision rules:\n"
        "1. A factual report without advice is Informational.\n"
        "2. A recommendation or suggested financial action is Advisory.\n"
        "3. An unresolved financial outcome presented as the main message, "
        "without a recommendation, is Uncertain.\n"
        "4. Do not use Uncertain merely because evidence is incomplete. "
        "Use the meaning and communicative purpose of the supplied text.\n\n"

        "Return a JSON object with exactly these keys: "
        '{"label":"Informational|Advisory|Uncertain",'
        '"explanation":"One concise sentence explaining the linguistic evidence."}'
    )


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "FinGuard API"}


@app.post("/api/classify", response_model=ClassificationResponse)
def classify(request: ClassificationRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=422, detail="Text cannot be empty.")

    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is missing from the backend environment.",
        )

    try:
        client = Groq()

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": build_prompt()},
                {
                    "role": "user",
                    "content": (
                        f"Context: {request.context}\n\n"
                        f"Text to classify: {text}"
                    ),
                },
            ],
            temperature=0,
        )

        raw = (response.choices[0].message.content or "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)

        label = result.get("label", "")
        explanation = result.get("explanation", "")

        if label not in LABELS or not isinstance(explanation, str):
            raise HTTPException(
                status_code=502,
                detail="The model returned an invalid classification response.",
            )

        return ClassificationResponse(
            label=label,
            explanation=explanation,
            model=MODEL,
        )

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=502,
            detail="The model returned an unreadable response. Please retry.",
        )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Classification failed. Check the API connection and try again.",
        )
