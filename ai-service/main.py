import os
import re
from datetime import datetime, timedelta
from typing import TypedDict, Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

load_dotenv()

app = FastAPI(title="WeatherGPT LangGraph AI Service")


# =========================================================
# CONFIG
# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
).strip()

ML_SERVICE_URL = os.getenv(
    "ML_SERVICE_URL",
    "http://localhost:8001"
).strip()


# =========================================================
# STATE
# =========================================================

class State(TypedDict, total=False):

    message: str

    city: str | None

    history: list[dict[str, Any]]

    intent: str

    location: dict[str, Any]

    weather: dict[str, Any]

    features: dict[str, float]

    risk: dict[str, Any]

    alert: str | None

    response: str

    target_date: str

    forecast_index: int


# =========================================================
# REQUEST MODEL
# =========================================================

class ChatRequest(BaseModel):

    message: str

    city: str | None = None

    # IMPORTANT:
    # Keep history flexible.
    #
    # The frontend may send forecastIndex as:
    # 1
    # "1"
    # null
    #
    # All are valid JSON values.
    history: list[dict[str, Any]] = Field(
        default_factory=list
    )


# =========================================================
# GEOCODING
# =========================================================

async def geocode(city: str):

    async with httpx.AsyncClient(
        timeout=10
    ) as client:

        response = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": city,
                "count": 1,
                "language": "en",
                "format": "json",
            },
        )

        response.raise_for_status()

        data = response.json()

        if not data.get("results"):

            raise ValueError(
                f"Location not found: {city}"
            )

        return data["results"][0]


# =========================================================
# WEATHER API
# =========================================================

async def weather_for(
    lat: float,
    lon: float
):

    params = {

        "latitude": lat,

        "longitude": lon,

        "current":
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "wind_speed_10m,"
            "surface_pressure",

        "daily":
            "temperature_2m_max,"
            "temperature_2m_min,"
            "precipitation_sum,"
            "precipitation_probability_max,"
            "weather_code,"
            "wind_speed_10m_max",

        "forecast_days": 7,

        "timezone": "auto",
    }


    async with httpx.AsyncClient(
        timeout=10
    ) as client:

        response = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params=params,
        )

        response.raise_for_status()

        return response.json()


# =========================================================
# QUERY UNDERSTANDING
# =========================================================

async def query_node(state: State):

    text = state["message"].lower().strip()


    # -----------------------------------------------------
    # Intent
    # -----------------------------------------------------

    if any(
        word in text
        for word in [
            "rain",
            "rainfall",
            "umbrella",
            "raining",
            "precipitation",
        ]
    ):

        intent = "RAIN_FORECAST"


    elif any(
        word in text
        for word in [
            "travel",
            "safe",
            "drive",
            "driving",
            "trip",
        ]
    ):

        intent = "TRAVEL_WEATHER"


    elif any(
        word in text
        for word in [
            "heat",
            "hot",
            "temperature",
            "cold",
        ]
    ):

        intent = "TEMPERATURE"


    elif any(
        word in text
        for word in [
            "storm",
            "thunder",
            "thunderstorm",
            "lightning",
        ]
    ):

        intent = "STORM"


    else:

        intent = "GENERAL_WEATHER"


    # -----------------------------------------------------
    # Date detection
    # -----------------------------------------------------

    today = datetime.now().date()

    forecast_index = 0


    # "day after tomorrow"
    if (
        "day after tomorrow" in text
        or "day after tmr" in text
        or "day after tom" in text
    ):

        forecast_index = 2


    # "tomorrow"
    elif "tomorrow" in text:

        forecast_index = 1


    # "today"
    elif "today" in text:

        forecast_index = 0


    # "three days from now"
    elif (
        "3 days from now" in text
        or "three days from now" in text
    ):

        forecast_index = 3


    # "four days from now"
    elif (
        "4 days from now" in text
        or "four days from now" in text
    ):

        forecast_index = 4


    # "five days from now"
    elif (
        "5 days from now" in text
        or "five days from now" in text
    ):

        forecast_index = 5


    # -----------------------------------------------------
    # Keep index inside 7-day forecast
    # -----------------------------------------------------

    forecast_index = max(
        0,
        min(
            forecast_index,
            6
        )
    )


    target_date = (
        today +
        timedelta(
            days=forecast_index
        )
    ).isoformat()


    return {

        "intent": intent,

        "forecast_index":
            forecast_index,

        "target_date":
            target_date,

    }


# =========================================================
# WEATHER NODE
# =========================================================

async def weather_node(state: State):

    city = (
        state.get("city")
        or "Nashik"
    )


    place = await geocode(city)


    weather = await weather_for(
        place["latitude"],
        place["longitude"]
    )


    return {

        "location": {

            "name":
                place["name"],

            "country":
                place.get("country"),

            "latitude":
                place["latitude"],

            "longitude":
                place["longitude"],

        },

        "weather":
            weather,

    }


# =========================================================
# RISK ANALYSIS
# =========================================================

async def risk_node(state: State):

    current = state["weather"]["current"]

    daily =state["weather"]["daily"]


    index = state.get(
        "forecast_index",
        1
    )


    # -----------------------------------------------------
    # Protect against forecast overflow
    # -----------------------------------------------------

    daily_length = len(
        daily.get(
            "time",
            []
        )
    )


    if daily_length == 0:

        index = 0

    else:

        index = min(
            index,
            daily_length - 1
        )


    # -----------------------------------------------------
    # Extract forecast values
    # -----------------------------------------------------

    rainfall = float(
        (
            daily.get(
                "precipitation_sum",
                [0]
            )[index]
        )
        or 0
    )


    rain_prob = float(
        (
            daily.get(
                "precipitation_probability_max",
                [0]
            )[index]
        )
        or 0
    )


    wind = float(
        (
            daily.get(
                "wind_speed_10m_max",
                [0]
            )[index]
        )
        or 0
    )


    temp = float(
        current.get(
            "temperature_2m",
            0
        )
        or 0
    )


    humidity = float(
        current.get(
            "relative_humidity_2m",
            0
        )
        or 0
    )


    pressure = float(
        current.get(
            "surface_pressure",
            1013
        )
        or 1013
    )


    features = {

        "temperature":
            temp,

        "humidity":
            humidity,

        "rainfall":
            rainfall,

        "rain_probability":
            rain_prob,

        "wind_speed":
            wind,

        "pressure":
            pressure,

    }


    # -----------------------------------------------------
    # Random Forest
    # -----------------------------------------------------

    async with httpx.AsyncClient(
        timeout=10
    ) as client:

        response = await client.post(

            f"{ML_SERVICE_URL}/predict",

            json=features,

        )

        response.raise_for_status()

        risk = response.json()


    # -----------------------------------------------------
    # Alert
    # -----------------------------------------------------

    alert = None


    if risk.get(
        "risk_level"
    ) == "High":

        alert = (

            f"High weather risk detected "
            f"for {state['location']['name']}. "
            f"Expected rainfall is about "
            f"{rainfall:.0f} mm with a "
            f"{rain_prob:.0f}% rain probability. "
            f"Check updated local advisories."
        )


    return {

        "features":
            features,

        "risk":
            risk,

        "alert":
            alert,

    }


# =========================================================
# GEMINI RESPONSE
# =========================================================

async def generate_gemini_response(
    state: State
):

    # -----------------------------------------------------
    # Fallback if Gemini key is missing
    # -----------------------------------------------------

    if not GEMINI_API_KEY:

        return None


    location =state["location"]["name"]

    features = state["features"]

    risk =state["risk"]

    intent =state.get(
            "intent",
            "GENERAL_WEATHER"
        )

    target_date =state.get(
            "target_date"
        )


    # -----------------------------------------------------
    # Conversation history
    # -----------------------------------------------------

    history_text = ""


    for item in state.get(
        "history",
        []
    ):

        role = str(
            item.get(
                "role",
                ""
            )
        )

        content = str(
            item.get(
                "content",
                ""
            )
        )


        if not content:
            continue


        if role == "user":

            history_text += (
                f"User: {content}\n"
            )


        elif role == "assistant":

            history_text += (
                f"WeatherGPT: {content}\n"
            )


    if not history_text:

        history_text = (
            "No previous conversation."
        )


    # -----------------------------------------------------
    # Grounded prompt
    # -----------------------------------------------------

    prompt = f"""

You are WeatherGPT, an AI weather assistant.

Answer the user's question using ONLY the
weather information supplied below.

Do not invent weather values.

Do not claim certainty beyond the forecast.

Be concise, natural and helpful.

If the user asks about rain, clearly mention
rain probability and expected rainfall.

If the user asks about temperature, mention
the relevant temperature.

If the user asks about travel, explain whether
conditions appear suitable based on the supplied
weather information.

If the risk is High, clearly mention the risk
and advise checking official local advisories.

The ML model is a weather-risk classifier.
It is NOT a replacement for official weather
forecasting.

Location:
{location}

Forecast date:
{target_date}

Intent:
{intent}

Temperature:
{features["temperature"]:.1f} °C

Humidity:
{features["humidity"]:.0f} %

Expected rainfall:
{features["rainfall"]:.1f} mm

Rain probability:
{features["rain_probability"]:.0f} %

Wind speed:
{features["wind_speed"]:.1f} km/h

Pressure:
{features["pressure"]:.1f} hPa

Risk level:
{risk.get("risk_level", "Unknown")}

Risk score:
{risk.get("risk_score", 0)}

Previous conversation:
{history_text}

Current user question:
{state["message"]}

Return ONLY the natural-language answer.
"""


    # -----------------------------------------------------
    # Gemini REST API
    # -----------------------------------------------------

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{GEMINI_MODEL}:generateContent"
    )


    payload = {

        "contents": [

            {

                "role": "user",

                "parts": [

                    {
                        "text": prompt
                    }

                ],

            }

        ]

    }


    headers = {

        "Content-Type":
            "application/json",

        "x-goog-api-key":
            GEMINI_API_KEY,

    }


    try:

        async with httpx.AsyncClient(
            timeout=30
        ) as client:

            response = await client.post(

                    url,

                    headers=headers,

                    json=payload,

                )


            response.raise_for_status()


            data =response.json()


        candidates =data.get(
                "candidates",
                []
            )


        if not candidates:

            return None


        parts = candidates[0].get(
                "content",
                {}
            ).get(
                "parts",
                []
            )


        text_parts = []


        for part in parts:

            if "text" in part:

                text_parts.append(
                    part["text"]
                )


        answer = "\n".join(
            text_parts
        ).strip()


        return answer or None


    except Exception as error:

        print(
            "Gemini error:",
            error
        )

        return None


# =========================================================
# RESPONSE NODE
# =========================================================

async def response_node(state: State):

    # -----------------------------------------------------
    # Try Gemini
    # -----------------------------------------------------

    gemini_answer =await generate_gemini_response(
            state
        )


    if gemini_answer:

        return {

            "response":
                gemini_answer

        }


    # -----------------------------------------------------
    # Safe fallback
    # -----------------------------------------------------

    location =state["location"]["name"]

    features = state["features"]

    risk =state["risk"].get(
            "risk_level",
            "Low"
        )


    if risk == "High":

        text = (

            f"High weather risk is "
            f"currently indicated for "
            f"{location}. The forecast for "
            f"{state.get('target_date')} "
            f"shows about "
            f"{features['rainfall']:.0f} mm "
            f"rainfall with a "
            f"{features['rain_probability']:.0f}% "
            f"rain probability. Check updated "
            f"local advisories before making "
            f"weather-sensitive plans."
        )


    elif risk == "Moderate":

        text = (

            f"Weather conditions in "
            f"{location} indicate moderate "
            f"weather risk for "
            f"{state.get('target_date')}. "
            f"Rain probability is around "
            f"{features['rain_probability']:.0f}% "
            f"with approximately "
            f"{features['rainfall']:.0f} mm "
            f"of expected rainfall."
        )


    else:

        text = (

            f"Weather conditions in "
            f"{location} on "
            f"{state.get('target_date')} "
            f"currently indicate low weather "
            f"risk. Rain probability is around "
            f"{features['rain_probability']:.0f}% "
            f"with approximately "
            f"{features['rainfall']:.0f} mm "
            f"of expected rainfall."
        )


    return {

        "response":
            text

    }


# =========================================================
# LANGGRAPH
# =========================================================

workflow = StateGraph(State)


workflow.add_node(
    "query_understanding",
    query_node
)

workflow.add_node(
    "weather_data",
    weather_node
)

workflow.add_node(
    "risk_analysis",
    risk_node
)

workflow.add_node(
    "response_generation",
    response_node
)


workflow.add_edge(
    START,
    "query_understanding"
)

workflow.add_edge(
    "query_understanding",
    "weather_data"
)

workflow.add_edge(
    "weather_data",
    "risk_analysis"
)

workflow.add_edge(
    "risk_analysis",
    "response_generation"
)

workflow.add_edge(
    "response_generation",
    END
)


graph = workflow.compile()


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    return {

        "ok": True,

        "langgraph": True,

        "gemini_configured":
            bool(GEMINI_API_KEY),

        "model":
            GEMINI_MODEL,

    }


# =========================================================
# CHAT
# =========================================================

@app.post("/chat")
async def chat(
    req: ChatRequest
):

    result = await graph.ainvoke({

        "message":
            req.message,

        "city":
            req.city,

        "history":
            req.history,

    })


    return {

        "response":
            result.get(
                "response",
                ""
            ),

        "location":
            result.get(
                "location"
            ),

        "weather":
            result.get(
                "weather"
            ),

        "risk":
            result.get(
                "risk"
            ),

        "alert":
            result.get(
                "alert"
            ),

        "intent":
            result.get(
                "intent"
            ),

        "target_date":
            result.get(
                "target_date"
            ),

        "forecast_index":
            result.get(
                "forecast_index"
            ),

        "features":
            result.get(
                "features"
            ),

        "timestamp":
            datetime.utcnow().isoformat()
            + "Z",

    }