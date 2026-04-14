"""FastAPI serving entrypoint for MDT model inference."""

from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config.configs import CognitiveArchitectureConfig, ModelConfig
from core.cognitive_streams import CognitiveStreamParser
from models.mdt_model import MedicalDigitalTwinModel


app = FastAPI(title="MDT Clinical AI API", version="1.0.0")


class ClinicalQuery(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_length: int = Field(default=512, ge=32, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class ClinicalResponse(BaseModel):
    response: str
    think_stream: str
    patient_state: str
    user_belief: str
    confidence: float


@lru_cache(maxsize=1)
def load_cached_model() -> MedicalDigitalTwinModel:
    config = ModelConfig()
    return MedicalDigitalTwinModel(config, use_demo_model=False)


@lru_cache(maxsize=1)
def load_parser() -> CognitiveStreamParser:
    return CognitiveStreamParser(CognitiveArchitectureConfig())


@app.get("/health")
def health() -> dict:
    return {"status": "healthy"}


@app.post("/generate", response_model=ClinicalResponse)
def generate_response(query: ClinicalQuery):
    try:
        model = load_cached_model()
        parser = load_parser()

        response = model.generate(
            query.prompt,
            max_length=query.max_length,
            temperature=query.temperature,
        )
        streams = parser.parse(response)

        stream_count = streams.count_complete_streams()
        confidence = float(stream_count / 3.0)

        return ClinicalResponse(
            response=response,
            think_stream=streams.think,
            patient_state=streams.patient_state,
            user_belief=streams.user_belief,
            confidence=confidence,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
