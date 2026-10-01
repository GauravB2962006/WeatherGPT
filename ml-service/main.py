from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

import joblib


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    Path(__file__).parent /
    "risk_model.joblib"
)

app = FastAPI(
    title="WeatherGPT ML Risk Service",
    version="4.0"
)


# ============================================================
# INPUT MODEL
# ============================================================

class WeatherFeatures(BaseModel):

    # Current-day weather
    temperature: float = Field(...)

    humidity: float = Field(
        ...,
        ge=0,
        le=100
    )

    rainfall: float = Field(
        ...,
        ge=0
    )

    wind_speed: float = Field(
        ...,
        ge=0
    )

    pressure: float = Field(...)

    # Kept for compatibility with the existing
    # WeatherGPT AI service.
    #
    # V4 does not use rain_probability directly.

    rain_probability: float = Field(
        ...,
        ge=0,
        le=100
    )

    # Previous-day weather

    previous_rainfall: float = Field(
        0.0,
        ge=0
    )

    previous_wind_speed: float = Field(
        0.0,
        ge=0
    )

    previous_humidity: float = Field(
        0.0,
        ge=0,
        le=100
    )

    # Seasonal features

    month: int = Field(
        ...,
        ge=1,
        le=12
    )

    day_of_year: int = Field(
        ...,
        ge=1,
        le=366
    )


# ============================================================
# LOAD MODEL
# ============================================================

def get_model():

    if not MODEL_PATH.exists():

        raise RuntimeError(
            "Model not trained. "
            "Run python train.py first."
        )

    return joblib.load(
        MODEL_PATH
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "ok": True,
        "model_ready": MODEL_PATH.exists(),
        "model_version": "V4"
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(
    features: WeatherFeatures
):

    model = get_model()


    # ========================================================
    # V4 FEATURE ORDER
    # ========================================================

    x = [[

        features.temperature,

        features.humidity,

        features.rainfall,

        features.wind_speed,

        features.pressure,

        features.previous_rainfall,

        features.previous_wind_speed,

        features.previous_humidity,

        features.month,

        features.day_of_year,

    ]]


    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = model.predict(x)[0]

    probabilities_array = (
        model.predict_proba(x)[0]
    )

    classes = list(
        model.classes_
    )


    # ========================================================
    # PROBABILITY DICTIONARY
    # ========================================================

    probabilities = {

        str(label):
            round(
                float(probability),
                3
            )

        for label, probability
        in zip(
            classes,
            probabilities_array
        )
    }


    # ========================================================
    # CONFIDENCE
    # ========================================================

    confidence = max(
        probabilities_array
    )


    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "risk_level":
            str(prediction),

        "risk_score":
            round(
                float(confidence),
                3
            ),

        "confidence":
            round(
                float(confidence),
                3
            ),

        "probabilities":
            probabilities,

        "model_version":
            "V4",

    }