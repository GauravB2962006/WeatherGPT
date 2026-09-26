# WeatherGPT MVP

SIH 2026 project: conversational weather intelligence with risk analysis, alerts, and LangGraph orchestration.

## Architecture

React/Vite frontend -> Node/Express backend -> Python FastAPI AI service -> LangGraph workflow -> Open-Meteo weather data + Random Forest risk model + optional Gemini response generation.

MongoDB and Firebase notification integration are prepared as extension points. The MVP uses a local JSON/SQLite-free fallback so it can run without a database during the first demo.

## Features in this MVP

- Current weather and forecast lookup by city
- Conversational weather questions
- LangGraph workflow with query understanding, weather retrieval, risk analysis, decision, and response generation
- Random Forest risk classifier with a bootstrap training dataset
- Low/Moderate/High risk output
- High-risk alert generation
- React dashboard with forecast cards and risk indicator
- Optional Gemini-powered natural-language responses
- Clear separation between verified weather data and AI-generated wording

## Prerequisites

- Node.js 20+
- Python 3.11+
- npm

## 1. Backend

```bash
cd backend
npm install
cp .env.example .env
npm run dev
```

Backend runs on `http://localhost:4000`.

## 2. ML service

```bash
cd ml-service
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train.py
uvicorn main:app --reload --port 8001
```

## 3. AI/LangGraph service

```bash
cd ai-service
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8002
```

The AI service can run without a Gemini key. In that case it uses a deterministic response template, while the LangGraph workflow and risk model still run.

## 4. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal, normally `http://localhost:5173`.

## Environment variables

Backend `.env`:

- `PORT=4000`
- `AI_SERVICE_URL=http://localhost:8002`

AI service `.env`:

- `GEMINI_API_KEY=` optional
- `ML_SERVICE_URL=http://localhost:8001`
- `WEATHER_TIMEOUT_SECONDS=10`

## Important SIH positioning

The MVP does **not** claim to replace professional meteorological forecasting. Weather values come from the weather data provider. The Random Forest model classifies potential risk from weather features. The LLM explains verified data and model output in natural language.

## Next implementation steps

1. Add MongoDB persistence for users, chat history, weather snapshots and alerts.
2. Add authentication and user location/preferences.
3. Add Leaflet maps and Recharts forecast graphs.
4. Add Firebase Cloud Messaging for push alerts.
5. Replace bootstrap training data with a real historical weather dataset.
6. Add multilingual responses (English/Hindi/Marathi).
7. Add evaluation, logging, rate limits, and deployment.
