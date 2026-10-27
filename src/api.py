from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import predict_text


app = FastAPI(
    title="Production Text Classification System",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
    )


class PredictionResponse(BaseModel):
    intent: str
    category: str
    confidence: float


@app.get("/health")
def health():
    return {
        "status": "ok",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(request: PredictionRequest):

    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=422,
            detail="Text cannot be empty",
        )

    return predict_text(text)