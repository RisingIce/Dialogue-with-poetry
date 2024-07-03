from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Union
import time


class RequestPoetry(BaseModel):
    query: str


class ResponsePoetry(BaseModel):
    response: Optional[str] = None


class RequestGenerate(BaseModel):
    type: str
    format: Optional[int] = 5
    emotion_type: Optional[int] = 1
    prompt: str


class RequestPairs(BaseModel):
    prompt: str
    predict_type: str
