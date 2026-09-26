import os
import re
import json
from datetime import datetime, date, timedelta
from typing import TypedDict, Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

load_dotenv()

app = FastAPI(title="WeatherGPT LangGraph AI Service")


# ============================================================
# CONFIGURATION
# ============================================================

ML_SERVICE_URL = os.getenv(
    "ML_SERVICE_URL",
    "http://localhost:8001"
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)


# ============================================================
# STATE
# ============================================================

class State(TypedDict, total=False):

    message: str
    city: str | None
    history: list[dict[str, Any]]

    # Query understanding
    scope: str
    query_type: str
    intent: str
    requested_city: str | None

    # Weather
    location: dict[str, Any]
    weather: dict[str, Any]

    target_date: str
    forecast_index: int

    # ML
    features: dict[str, float]
    risk: dict[str, Any]

    alert: str | None

    # Final answer
    response: str

    # Frontend helper
    is_weather_related: bool


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):

    message: str

    city: str | None = None

    history: list[dict[str, Any]] = Field(
        default_factory=list
    )


# ============================================================
# GEMINI HELPER
# ============================================================

async def call_gemini(prompt: str):

    if not GEMINI_API_KEY:
        return None

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/{GEMINI_MODEL}:generateContent"
    )

    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "Content-Type": "application/json"
    }

    body = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }

    try:

        async with httpx.AsyncClient(timeout=30) as client:

            response = await client.post(
                url,
                headers=headers,
                json=body
            )

            response.raise_for_status()

            data = response.json()

        candidates = data.get("candidates", [])

        if not candidates:
            return None

        parts = (
            candidates[0]
            .get("content", {})
            .get("parts", [])
        )

        if not parts:
            return None

        return parts[0].get("text", "").strip()

    except Exception as error:

        print("Gemini error:", str(error))

        return None


# ============================================================
# FALLBACK WEATHER KEYWORDS
# ============================================================

WEATHER_KEYWORDS = [
    "weather",
    "forecast",
    "rain",
    "rainfall",
    "raining",
    "umbrella",
    "precipitation",
    "shower",
    "storm",
    "thunderstorm",
    "lightning",
    "cloud",
    "cloudy",
    "sunny",
    "sunshine",
    "temperature",
    "temp",
    "hot",
    "heat",
    "cold",
    "cool",
    "humidity",
    "humid",
    "wind",
    "windy",
    "breeze",
    "pressure",
    "atmosphere",
    "climate",
    "monsoon",
    "cyclone",
    "tornado",
    "hurricane",
    "snow",
    "snowfall",
    "fog",
    "mist",
    "hail",
    "drought",
    "flood",
    "heatwave",
    "heat wave",
    "coldwave",
    "cold wave",
    "dew",
    "visibility",
    "uv",
    "uv index",
    "air pressure",
    "weather alert",
    "weather warning",
    "outdoor",
    "outside",
    "travel",
    "trip",
    "journey",
    "drive",
    "driving",
    "bike ride",
    "cycling",
    "picnic",
    "walk",
    "walking",
    "commute",
    "commuting"
]


# ============================================================
# HEURISTIC WEATHER DETECTION
# ============================================================

def heuristic_weather_detection(text: str):

    lower = text.lower()

    if any(
        keyword in lower
        for keyword in WEATHER_KEYWORDS
    ):
        return True

    return False


# ============================================================
# QUERY UNDERSTANDING
# ============================================================

async def query_node(state: State):

    message = state["message"].strip()

    lower = message.lower()

    history = state.get("history", [])

    # --------------------------------------------------------
    # First: use Gemini for broad semantic understanding
    # --------------------------------------------------------

    if GEMINI_API_KEY:

        history_text = ""

        for item in history[-6:]:

            role = item.get("role", "")

            content = item.get("content", "")

            history_text += (
                f"{role}: {content}\n"
            )

        prompt = f"""
You are the query-understanding system for WeatherGPT.

WeatherGPT is ONLY for weather and weather-related information.

Classify the user's question.

WEATHER-RELATED QUESTIONS INCLUDE:

- Current weather
- Weather forecasts
- Rain
- Rain probability
- Temperature
- Humidity
- Wind
- Storms
- Thunderstorms
- Lightning
- Clouds
- Sunshine
- Snow
- Fog
- Visibility
- Pressure
- UV
- Monsoon
- Cyclones
- Hurricanes
- Tornadoes
- Floods caused by weather
- Heat waves
- Cold waves
- Climate/weather concepts
- Weather science
- Weather safety
- Outdoor activities affected by weather
- Travel affected by weather
- Bike rides affected by weather
- Farming questions where weather is relevant
- What to wear based on weather
- Weather comparisons
- Historical/current/future weather questions

NON-WEATHER QUESTIONS INCLUDE:

- Programming
- Java
- C/C++
- Mathematics
- General knowledge unrelated to weather
- Movies
- Sports unrelated to weather
- Politics
- Celebrities
- Entertainment
- Coding help
- Homework unrelated to weather
- General conversation
- Jokes
- Anything unrelated to weather

IMPORTANT:

A question such as:
"Can I go for a bike ride tomorrow?"

IS WEATHER-RELATED because weather affects the activity.

A question such as:
"What causes thunderstorms?"

IS WEATHER-RELATED even though it is not asking for a forecast.

A question such as:
"Who is the Prime Minister?"

IS NOT WEATHER-RELATED.

Return ONLY valid JSON.

JSON FORMAT:

{{
    "scope": "weather" or "non_weather",
    "query_type": "forecast" or "weather_knowledge",
    "intent": "short descriptive intent",
    "requested_city": "city name or null"
}}

USER QUESTION:
{message}

RECENT CONVERSATION:
{history_text}
"""

        result = await call_gemini(prompt)

        if result:

            try:

                cleaned = result.strip()

                # Remove markdown JSON fences if Gemini adds them
                cleaned = re.sub(
                    r"^```json\s*",
                    "",
                    cleaned,
                    flags=re.IGNORECASE
                )

                cleaned = re.sub(
                    r"\s*```$",
                    "",
                    cleaned
                )

                parsed = json.loads(cleaned)

                scope = parsed.get(
                    "scope",
                    "weather"
                )

                query_type = parsed.get(
                    "query_type",
                    "forecast"
                )

                intent = parsed.get(
                    "intent",
                    "GENERAL_WEATHER"
                )

                requested_city = parsed.get(
                    "requested_city"
                )

                if scope not in [
                    "weather",
                    "non_weather"
                ]:
                    scope = "weather"

                if query_type not in [
                    "forecast",
                    "weather_knowledge"
                ]:
                    query_type = "forecast"

                return {
                    "scope": scope,
                    "query_type": query_type,
                    "intent": intent.upper(),
                    "requested_city": requested_city,
                    "is_weather_related": (
                        scope == "weather"
                    )
                }

            except Exception as error:

                print(
                    "Query classification error:",
                    str(error)
                )

    # --------------------------------------------------------
    # FALLBACK CLASSIFICATION
    # --------------------------------------------------------

    is_weather = heuristic_weather_detection(
        lower
    )

    if not is_weather:

        return {
            "scope": "non_weather",
            "query_type": "weather_knowledge",
            "intent": "OUT_OF_SCOPE",
            "requested_city": None,
            "is_weather_related": False
        }

    # Most weather questions without explicit
    # forecasting words are treated as knowledge.
    forecast_words = [
        "today",
        "tomorrow",
        "forecast",
        "will",
        "next",
        "this week",
        "this weekend",
        "day after",
        "weather now",
        "right now",
        "current weather",
        "temperature today",
        "temperature tomorrow"
    ]

    query_type = (
        "forecast"
        if any(word in lower for word in forecast_words)
        else "weather_knowledge"
    )

    return {
        "scope": "weather",
        "query_type": query_type,
        "intent": "GENERAL_WEATHER",
        "requested_city": None,
        "is_weather_related": True
    }


# ============================================================
# DATE UNDERSTANDING
# ============================================================

def resolve_weekday(
    text: str,
    reference_date: date
):

    weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6
    }

    for name, weekday_number in weekdays.items():

        if re.search(
            rf"\b{name}\b",
            text
        ):

            days_ahead = (
                weekday_number
                - reference_date.weekday()
            ) % 7

            # If today is the requested weekday,
            # interpret it as today.
            return days_ahead

    return None


def get_previous_target_date(history):

    for item in reversed(history):

        target = item.get("targetDate")

        if target:
            try:
                return date.fromisoformat(target)
            except:
                pass

    return None


def get_forecast_index(
    message: str,
    available_dates: list[str],
    history: list[dict[str, Any]]
):

    text = message.lower().strip()

    today = date.today()

    # --------------------------------------------------------
    # TODAY
    # --------------------------------------------------------

    if re.search(r"\btoday\b", text):

        return 0

    # --------------------------------------------------------
    # DAY AFTER TOMORROW
    # --------------------------------------------------------

    if (
        "day after tomorrow" in text
        or "after tomorrow" in text
        or "day after tmr" in text
    ):

        return 2

    # --------------------------------------------------------
    # TOMORROW
    # --------------------------------------------------------

    if (
        re.search(r"\btomorrow\b", text)
        or re.search(r"\btmrw\b|\btmr\b", text)
    ):

        return 1

    # --------------------------------------------------------
    # IN N DAYS
    # --------------------------------------------------------

    number_words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7
    }

    match = re.search(
        r"\bin\s+(\d+|one|two|three|four|five|six|seven)\s+days?\b",
        text
    )

    if match:

        value = match.group(1)

        days = (
            int(value)
            if value.isdigit()
            else number_words[value]
        )

        return days

    # --------------------------------------------------------
    # FOLLOW-UP: DAY AFTER / NEXT DAY
    # --------------------------------------------------------

    previous_target = get_previous_target_date(
        history
    )

    if previous_target:

        if any(
            phrase in text
            for phrase in [
                "day after",
                "next day",
                "following day",
                "day after that",
                "what about the next"
            ]
        ):

            difference = (
                previous_target - today
            ).days

            return max(
                difference + 1,
                0
            )

        if any(
            phrase in text
            for phrase in [
                "that day",
                "same day",
                "that date",
                "what about that",
                "how about that",
                "then"
            ]
        ):

            difference = (
                previous_target - today
            ).days

            return max(
                difference,
                0
            )

        # Generic follow-up
        if any(
            text.startswith(prefix)
            for prefix in [
                "what about",
                "how about",
                "and "
            ]
        ):

            difference = (
                previous_target - today
            ).days

            return max(
                difference,
                0
            )

    # --------------------------------------------------------
    # WEEKDAY
    # --------------------------------------------------------

    weekday_index = resolve_weekday(
        text,
        today
    )

    if weekday_index is not None:

        return weekday_index

    # --------------------------------------------------------
    # DEFAULT
    #
    # For forecast questions without a date,
    # use tomorrow.
    # --------------------------------------------------------

    return 1


# ============================================================
# WEATHER API
# ============================================================

async def geocode(city: str):

    async with httpx.AsyncClient(timeout=10) as client:

        response = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": city,
                "count": 1,
                "language": "en",
                "format": "json"
            }
        )

        response.raise_for_status()

        data = response.json()

        if not data.get("results"):

            raise ValueError(
                f"Location not found: {city}"
            )

        return data["results"][0]


async def weather_for(
    lat: float,
    lon: float
):

    params = {

        "latitude": lat,

        "longitude": lon,

        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "wind_speed_10m,"
            "surface_pressure"
        ),

        "daily": (
            "temperature_2m_max,"
            "temperature_2m_min,"
            "precipitation_sum,"
            "precipitation_probability_max,"
            "weather_code,"
            "wind_speed_10m_max"
        ),

        "forecast_days": 7,

        "timezone": "auto"
    }

    async with httpx.AsyncClient(timeout=10) as client:

        response = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params=params
        )

        response.raise_for_status()

        return response.json()


# ============================================================
# WEATHER DATA NODE
# ============================================================

async def weather_node(state: State):

    requested_city = state.get(
        "requested_city"
    )

    city = (
        requested_city
        or state.get("city")
        or "Nashik"
    )

    place = await geocode(city)

    data = await weather_for(
        place["latitude"],
        place["longitude"]
    )

    available_dates = data["daily"]["time"]

    requested_index = get_forecast_index(
        state["message"],
        available_dates,
        state.get("history", [])
    )

    forecast_index = min(
        max(requested_index, 0),
        len(available_dates) - 1
    )

    target_date = available_dates[
        forecast_index
    ]

    location = {

        "name": place["name"],

        "country": place.get("country"),

        "latitude": place["latitude"],

        "longitude": place["longitude"]
    }

    return {

        "location": location,

        "weather": data,

        "target_date": target_date,

        "forecast_index": forecast_index
    }


# ============================================================
# RISK ANALYSIS
# ============================================================

async def risk_node(state: State):

    weather = state["weather"]

    daily = weather["daily"]

    index = state["forecast_index"]

    rainfall = float(
        daily["precipitation_sum"][index] or 0
    )

    rain_probability = float(
        daily["precipitation_probability_max"][index]
        or 0
    )

    wind = float(
        daily["wind_speed_10m_max"][index] or 0
    )

    temperature = float(
        daily["temperature_2m_max"][index] or 0
    )

    current = weather.get(
        "current",
        {}
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

        "temperature": temperature,

        "humidity": humidity,

        "rainfall": rainfall,

        "rain_probability": rain_probability,

        "wind_speed": wind,

        "pressure": pressure
    }

    try:

        async with httpx.AsyncClient(
            timeout=10
        ) as client:

            response = await client.post(
                f"{ML_SERVICE_URL}/predict",
                json=features
            )

            response.raise_for_status()

            risk = response.json()

    except Exception as error:

        print(
            "ML service error:",
            str(error)
        )

        # Safe fallback if ML service is temporarily unavailable.
        if (
            rain_probability >= 70
            or rainfall >= 20
            or wind >= 45
        ):
            level = "High"

        elif (
            rain_probability >= 40
            or rainfall >= 5
            or wind >= 30
        ):
            level = "Medium"

        else:
            level = "Low"

        risk = {
            "risk_level": level,
            "source": "fallback"
        }

    alert = None

    if risk.get("risk_level") == "High":

        alert = (
            f"High weather risk detected for "
            f"{state['location']['name']} on "
            f"{state['target_date']}. "
            f"Expected rainfall is about "
            f"{rainfall:.0f} mm with a "
            f"{rain_probability:.0f}% rain probability. "
            f"Check updated local weather advisories."
        )

    return {

        "features": features,

        "risk": risk,

        "alert": alert
    }


# ============================================================
# OUT-OF-SCOPE RESPONSE
# ============================================================

def out_of_scope_response():

    return (
        "I'm WeatherGPT, and this website is designed "
        "specifically for weather-related information. "
        "I can help with forecasts, rain, temperature, "
        "wind, storms, climate, weather safety, travel "
        "conditions and other weather-related questions."
    )


# ============================================================
# WEATHER KNOWLEDGE RESPONSE
# ============================================================

async def generate_weather_knowledge_response(
    state: State
):

    prompt = f"""
You are WeatherGPT, an AI assistant specialized ONLY in weather.

Answer the user's weather-related question clearly.

This may be a conceptual weather question rather than
a forecast question.

Examples:

- What causes thunderstorms?
- What is El Niño?
- How does monsoon work?
- Why does humidity increase?
- What is atmospheric pressure?
- How are cyclones formed?

USER QUESTION:
{state["message"]}

RULES:

1. Answer only the weather-related question.
2. Do not invent current weather data.
3. Do not provide today's or tomorrow's weather unless
   the user explicitly asks for a forecast.
4. Do not force a forecast into the answer.
5. Keep the explanation simple.
6. Use examples when helpful.
7. If the question is ambiguous, explain the weather-related
   interpretation.
8. Keep the answer around 3-6 sentences.
"""

    result = await call_gemini(prompt)

    if result:

        return result

    return (
        "That is a weather-related topic. "
        "Please make sure your Gemini API is configured "
        "to receive a detailed AI explanation."
    )


# ============================================================
# FORECAST RESPONSE
# ============================================================

async def generate_forecast_response(
    state: State
):

    location = state["location"]["name"]

    target_date = state["target_date"]

    features = state["features"]

    risk_level = state["risk"].get(
        "risk_level",
        "Low"
    )

    rainfall = features["rainfall"]

    rain_probability = features[
        "rain_probability"
    ]

    temperature = features[
        "temperature"
    ]

    wind = features[
        "wind_speed"
    ]

    humidity = features[
        "humidity"
    ]

    pressure = features[
        "pressure"
    ]

    prompt = f"""
You are WeatherGPT, a conversational weather assistant.

Answer the user's actual question using the supplied
weather information.

USER QUESTION:
{state["message"]}

LOCATION:
{location}

FORECAST DATE:
{target_date}

WEATHER DATA:
Temperature: {temperature:.1f} °C
Rain probability: {rain_probability:.0f} %
Expected rainfall: {rainfall:.1f} mm
Wind speed: {wind:.1f} km/h
Humidity: {humidity:.0f} %
Pressure: {pressure:.1f} hPa

MACHINE LEARNING WEATHER RISK:
{risk_level}

RULES:

1. Answer the user's actual question directly.

2. Use ONLY the supplied weather values.

3. Do not invent weather information.

4. Do not answer with generic "today's weather"
   unless the user asked about today.

5. Do not answer with tomorrow's weather unless
   the question is actually about tomorrow.

6. If the question is about rain, focus on:
   rain probability and expected rainfall.

7. If the question is about temperature, focus on:
   temperature.

8. If the question is about wind, focus on:
   wind speed.

9. If the question involves travel, cycling, walking,
   outdoor activities or safety, explain how the supplied
   weather conditions affect the activity.

10. If the question asks for a recommendation such as
    whether an activity is suitable, base the explanation
    on the weather conditions and ML risk level.
    Do not pretend the ML model is a professional
    meteorological warning system.

11. If risk is High, recommend checking current local
    weather advisories.

12. Mention the forecast date when useful.

13. Keep the response natural and concise.

14. Use previous conversation context when needed to
    understand follow-up questions.
"""

    # Add conversation history
    history = state.get(
        "history",
        []
    )

    if history:

        history_text = "\n".join(
            [
                f"{item.get('role')}: {item.get('content')}"
                for item in history[-6:]
            ]
        )

        prompt += f"""

RECENT CONVERSATION:

{history_text}

Use this only to understand context.
Do not copy previous answers blindly.
"""

    result = await call_gemini(prompt)

    if result:

        return result

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    intent = state.get(
        "intent",
        "GENERAL_WEATHER"
    )

    if intent == "RAIN_FORECAST":

        return (
            f"For {target_date} in {location}, "
            f"the rain probability is around "
            f"{rain_probability:.0f}% with expected "
            f"rainfall of about {rainfall:.0f} mm."
        )

    if intent == "TEMPERATURE":

        return (
            f"On {target_date} in {location}, "
            f"the forecast high is around "
            f"{temperature:.1f}°C."
        )

    if intent == "WIND":

        return (
            f"On {target_date} in {location}, "
            f"maximum wind speed is expected to be "
            f"around {wind:.1f} km/h."
        )

    return (
        f"For {target_date} in {location}, "
        f"the weather risk is {risk_level.lower()}. "
        f"Rain probability is around "
        f"{rain_probability:.0f}% and expected rainfall "
        f"is approximately {rainfall:.0f} mm."
    )


# ============================================================
# RESPONSE NODE
# ============================================================

async def response_node(state: State):

    scope = state.get(
        "scope",
        "weather"
    )

    query_type = state.get(
        "query_type",
        "forecast"
    )

    # --------------------------------------------------------
    # NON-WEATHER
    # --------------------------------------------------------

    if scope == "non_weather":

        return {

            "response": out_of_scope_response(),

            "is_weather_related": False
        }

    # --------------------------------------------------------
    # WEATHER KNOWLEDGE
    # --------------------------------------------------------

    if query_type == "weather_knowledge":

        response = (
            await generate_weather_knowledge_response(
                state
            )
        )

        return {

            "response": response,

            "is_weather_related": True
        }

    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    response = await generate_forecast_response(
        state
    )

    return {

        "response": response,

        "is_weather_related": True
    }


# ============================================================
# LANGGRAPH WORKFLOW
# ============================================================

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


# ------------------------------------------------------------
# IMPORTANT:
#
# Non-weather questions skip weather API + ML completely.
#
# Weather knowledge questions also skip forecast + ML.
# ------------------------------------------------------------

def route_after_query(state: State):

    if state.get("scope") == "non_weather":

        return "response_generation"

    if state.get("query_type") == "weather_knowledge":

        return "response_generation"

    return "weather_data"


workflow.add_conditional_edges(
    "query_understanding",
    route_after_query,
    {
        "weather_data": "weather_data",
        "response_generation": "response_generation"
    }
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


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {

        "ok": True,

        "langgraph": True,

        "gemini_configured": bool(
            GEMINI_API_KEY
        ),

        "model": GEMINI_MODEL
    }


# ============================================================
# CHAT ENDPOINT
# ============================================================

@app.post("/chat")
async def chat(req: ChatRequest):

    result = await graph.ainvoke(
        {
            "message": req.message,

            "city": req.city,

            "history": req.history
        }
    )

    return {

        "response": result.get(
            "response",
            ""
        ),

        "location": result.get(
            "location"
        ),

        "weather": result.get(
            "weather"
        ),

        "risk": result.get(
            "risk"
        ),

        "alert": result.get(
            "alert"
        ),

        "intent": result.get(
            "intent"
        ),

        "target_date": result.get(
            "target_date"
        ),

        "forecast_index": result.get(
            "forecast_index"
        ),

        "features": result.get(
            "features"
        ),

        "is_weather_related": result.get(
            "is_weather_related",
            True
        ),

        "query_type": result.get(
            "query_type"
        ),

        "timestamp": (
            datetime.utcnow().isoformat()
            + "Z"
        )
    }