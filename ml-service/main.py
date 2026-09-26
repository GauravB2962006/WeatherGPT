from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib

MODEL_PATH = Path(__file__).parent / 'risk_model.joblib'
app = FastAPI(title='WeatherGPT ML Risk Service')

class WeatherFeatures(BaseModel):
    temperature: float = Field(...)
    humidity: float = Field(..., ge=0, le=100)
    rainfall: float = Field(..., ge=0)
    rain_probability: float = Field(..., ge=0, le=100)
    wind_speed: float = Field(..., ge=0)
    pressure: float = Field(...)


def get_bundle():
    if not MODEL_PATH.exists():
        raise RuntimeError('Model not trained. Run python train.py first.')
    return joblib.load(MODEL_PATH)

@app.get('/health')
def health():
    return {'ok': True, 'model_ready': MODEL_PATH.exists()}

@app.post('/predict')
def predict(features: WeatherFeatures):
    bundle = get_bundle()
    x = [[features.temperature, features.humidity, features.rainfall, features.rain_probability, features.wind_speed, features.pressure]]
    pred = int(bundle['model'].predict(x)[0])
    probabilities = bundle['model'].predict_proba(x)[0]
    return {'risk_level': bundle['labels'][pred], 'risk_score': round(float(max(probabilities)), 3), 'probabilities': [round(float(p), 3) for p in probabilities]}
