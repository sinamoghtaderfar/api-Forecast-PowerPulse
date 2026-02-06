# app/schemas.py
from pydantic import BaseModel
from typing import List

class ForecastItem(BaseModel):
    year: int
    forecast: float
    lower: float | None = None  
    upper: float | None = None

class ForecastResponse(BaseModel):
    prophet: List[ForecastItem]
    random_forest: List[ForecastItem]
    xgboost: List[ForecastItem]

class CombinedForecastResponse(BaseModel):
    combined: List[ForecastItem]
