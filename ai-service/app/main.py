import os
import re
import json
import asyncio
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
    ml_features: dict[str, float]
    risk: dict[str, Any]

    alert: str | None

    # Response
    response: str

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
# GEMINI
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

    max_attempts = 3

    for attempt in range(max_attempts):

        try:

            async with httpx.AsyncClient(
                timeout=30
            ) as client:

                response = await client.post(
                    url,
                    headers=headers,
                    json=body
                )

            # ------------------------------------------------
            # RATE LIMIT
            # ------------------------------------------------

            if response.status_code == 429:

                # Do not retry an exhausted/rate-limited Gemini
                # request. Return immediately so WeatherGPT can
                # use its deterministic fallback response.
                print(
                    "Gemini rate limited (429). "
                    "Using fallback response."
                )

                return None

            # ------------------------------------------------
            # TEMPORARY SERVER ERRORS
            # ------------------------------------------------

            if response.status_code in [
                500,
                502,
                503,
                504
            ]:

                if attempt < max_attempts - 1:

                    wait_time = 2 ** attempt

                    print(
                        f"Gemini server error "
                        f"{response.status_code}. "
                        f"Retrying in {wait_time}s..."
                    )

                    await asyncio.sleep(
                        wait_time
                    )

                    continue

            response.raise_for_status()

            data = response.json()

            candidates = data.get(
                "candidates",
                []
            )

            if not candidates:
                return None

            parts = (
                candidates[0]
                .get("content", {})
                .get("parts", [])
            )

            if not parts:
                return None

            text = parts[0].get("text")

            if text:
                return text.strip()

            return None

        except httpx.TimeoutException as error:

            if attempt < max_attempts - 1:

                wait_time = 2 ** attempt

                print(
                    f"Gemini timeout. "
                    f"Retrying in {wait_time}s..."
                )

                await asyncio.sleep(
                    wait_time
                )

                continue

            print(
                "Gemini timeout:",
                str(error)
            )

        except httpx.RequestError as error:

            if attempt < max_attempts - 1:

                wait_time = 2 ** attempt

                print(
                    f"Gemini connection error. "
                    f"Retrying in {wait_time}s..."
                )

                await asyncio.sleep(
                    wait_time
                )

                continue

            print(
                "Gemini request error:",
                str(error)
            )

        except Exception as error:

            print(
                "Gemini error:",
                str(error)
            )

            return None

    return None


# ============================================================
# WEATHER KEYWORDS
# ============================================================

WEATHER_KEYWORDS = [

    "weather",
    "forecast",

    "rain",
    "rainfall",
    "raining",
    "rainy",
    "umbrella",
    "precipitation",
    "shower",

    "storm",
    "storms",
    "thunderstorm",
    "thunderstorms",
    "lightning",

    "cloud",
    "clouds",
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
    "atmospheric pressure",

    "climate",
    "monsoon",

    "cyclone",
    "cyclones",
    "hurricane",
    "hurricanes",
    "tornado",
    "tornadoes",

    "snow",
    "snowfall",

    "fog",
    "mist",

    "hail",

    "drought",
    "flood",
    "flooding",

    "heatwave",
    "heat wave",

    "coldwave",
    "cold wave",

    "dew",

    "visibility",

    "uv",
    "uv index",

    "weather alert",
    "weather warning",

    # Weather-dependent activities
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
# WEATHER DETECTION FALLBACK
# ============================================================

def looks_weather_related(text: str):

    lower = text.lower()

    return any(
        keyword in lower
        for keyword in WEATHER_KEYWORDS
    )


# ============================================================
# QUERY UNDERSTANDING
# ============================================================

def extract_requested_city(message: str):

    patterns = [
        r"\bweather\s+(?:in|at|for)\s+([A-Za-z][A-Za-z .'-]{1,40}?)(?:\?|$)",
        r"\bforecast\s+(?:in|at|for)\s+([A-Za-z][A-Za-z .'-]{1,40}?)(?:\?|$)",
        r"\b(?:rain|temperature|wind|weather)\s+(?:in|at)\s+([A-Za-z][A-Za-z .'-]{1,40}?)(?:\?|$)"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            message,
            flags=re.IGNORECASE
        )

        if match:

            city = match.group(1).strip()

            city = re.sub(
                r"\s+(today|tomorrow|tonight|now)$",
                "",
                city,
                flags=re.IGNORECASE
            ).strip()

            if city:
                return city

    return None


# ============================================================
# LOCAL QUERY CLASSIFICATION
# ============================================================

def local_query_classification(
    message: str,
    history: list[dict[str, Any]]
):
    """
    Classify obvious questions locally so Gemini is not
    unnecessarily called for every message.
    """

    text = message.lower().strip()

    requested_city = extract_requested_city(message)

    # --------------------------------------------------------
    # OBVIOUS NON-WEATHER
    # --------------------------------------------------------

    if not looks_weather_related(text):

        follow_up_phrases = [
            "what about",
            "how about",
            "what about that",
            "how about that",
            "that day",
            "same day",
            "that date",
            "next day",
            "following day",
            "day after that",
            "what about the next",
            "then"
        ]

        has_follow_up = any(
            phrase in text
            for phrase in follow_up_phrases
        )

        if has_follow_up and history:

            return {
                "scope": "weather",
                "query_type": "forecast",
                "intent": "GENERAL_WEATHER",
                "requested_city": None,
                "is_weather_related": True
            }

        return {
            "scope": "non_weather",
            "query_type": "weather_knowledge",
            "intent": "OUT_OF_SCOPE",
            "requested_city": None,
            "is_weather_related": False
        }

    # --------------------------------------------------------
    # WEATHER KNOWLEDGE QUESTIONS
    # --------------------------------------------------------

    knowledge_words = [
        "what is",
        "what are",
        "why does",
        "why do",
        "why is",
        "why are",
        "how does",
        "how do",
        "how is",
        "how are",
        "causes",
        "cause of",
        "meaning of",
        "explain",
        "difference between",
        "how works",
        "how does it work"
    ]

    if any(
        phrase in text
        for phrase in knowledge_words
    ):

        return {
            "scope": "weather",
            "query_type": "weather_knowledge",
            "intent": "WEATHER_KNOWLEDGE",
            "requested_city": requested_city,
            "is_weather_related": True
        }

    # --------------------------------------------------------
    # FORECAST / ACTUAL WEATHER DATA
    # --------------------------------------------------------

    forecast_words = [
        "today",
        "tomorrow",
        "tmrw",
        "tmr",
        "forecast",
        "current",
        "now",
        "will",
        "next",
        "this week",
        "this weekend",
        "day after",
        "yesterday",
        "tonight",
        "weather",
        "weather in",
        "weather for",
        "weather at",
        "temperature today",
        "temperature tomorrow",
        "weather today",
        "weather tomorrow"
    ]

    if any(
        word in text
        for word in forecast_words
    ):

        return {
            "scope": "weather",
            "query_type": "forecast",
            "intent": "GENERAL_WEATHER",
            "requested_city": requested_city,
            "is_weather_related": True
        }

    # --------------------------------------------------------
    # RAIN
    # --------------------------------------------------------

    if any(
        word in text
        for word in [
            "rain",
            "rainfall",
            "raining",
            "rainy",
            "umbrella",
            "precipitation",
            "shower"
        ]
    ):

        return {
            "scope": "weather",
            "query_type": "forecast",
            "intent": "RAIN_FORECAST",
            "requested_city": requested_city,
            "is_weather_related": True
        }

    # --------------------------------------------------------
    # TEMPERATURE
    # --------------------------------------------------------

    if any(
        word in text
        for word in [
            "temperature",
            "temp",
            "hot",
            "cold",
            "heat",
            "cool"
        ]
    ):

        return {
            "scope": "weather",
            "query_type": "forecast",
            "intent": "TEMPERATURE",
            "requested_city": requested_city,
            "is_weather_related": True
        }

    # --------------------------------------------------------
    # WIND
    # --------------------------------------------------------

    if any(
        word in text
        for word in [
            "wind",
            "windy",
            "breeze"
        ]
    ):

        return {
            "scope": "weather",
            "query_type": "forecast",
            "intent": "WIND",
            "requested_city": requested_city,
            "is_weather_related": True
        }

    # --------------------------------------------------------
    # OUTDOOR / TRAVEL ACTIVITIES
    # --------------------------------------------------------

    if any(
        word in text
        for word in [
            "bike ride",
            "cycling",
            "travel",
            "trip",
            "journey",
            "drive",
            "driving",
            "walk",
            "walking",
            "picnic",
            "outdoor",
            "outside",
            "commute",
            "commuting"
        ]
    ):

        return {
            "scope": "weather",
            "query_type": "forecast",
            "intent": "GENERAL_WEATHER",
            "requested_city": requested_city,
            "is_weather_related": True
        }

    # --------------------------------------------------------
    # OTHER CLEAR WEATHER KNOWLEDGE TOPICS
    # --------------------------------------------------------

    if any(
        word in text
        for word in [
            "humidity",
            "humid",
            "storm",
            "storms",
            "thunderstorm",
            "thunderstorms",
            "lightning",
            "cloud",
            "clouds",
            "cloudy",
            "sunny",
            "sunshine",
            "snow",
            "snowfall",
            "fog",
            "mist",
            "hail",
            "climate",
            "monsoon",
            "cyclone",
            "cyclones",
            "hurricane",
            "hurricanes",
            "tornado",
            "tornadoes",
            "flood",
            "flooding",
            "drought",
            "heatwave",
            "heat wave",
            "coldwave",
            "cold wave",
            "visibility",
            "uv",
            "uv index",
            "pressure",
            "atmospheric pressure",
            "weather alert",
            "weather warning"
        ]
    ):

        return {
            "scope": "weather",
            "query_type": "weather_knowledge",
            "intent": "WEATHER_KNOWLEDGE",
            "requested_city": requested_city,
            "is_weather_related": True
        }

    return None


# ============================================================
# QUERY UNDERSTANDING
# ============================================================

async def query_node(state: State):

    message = state["message"].strip()

    history = state.get(
        "history",
        []
    )

    # ========================================================
    # STEP 1 — LOCAL CLASSIFICATION
    # ========================================================

    local_result = local_query_classification(
        message,
        history
    )

    if local_result is not None:

        print(
            "Local classifier used:",
            local_result["scope"],
            local_result["query_type"],
            local_result["intent"]
        )

        return local_result

    # ========================================================
    # STEP 2 — GEMINI ONLY FOR AMBIGUOUS QUESTIONS
    # ========================================================

    print(
        "Gemini classifier used for ambiguous query."
    )

    if GEMINI_API_KEY:

        history_text = "\n".join(
            [
                f"{item.get('role', '')}: "
                f"{item.get('content', '')}"
                for item in history[-6:]
            ]
        )

        prompt = f"""
You are the query classifier for WeatherGPT.

WeatherGPT is a specialized weather assistant.

Your job is to decide whether the user's question is
related to WEATHER.

Weather-related questions include:

- Current weather
- Weather forecasts
- Rain
- Rain probability
- Rainfall
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
- Atmospheric pressure
- UV
- Monsoon
- Cyclones
- Hurricanes
- Tornadoes
- Floods related to weather
- Drought
- Heat waves
- Cold waves
- Climate
- Weather science
- Weather safety
- Outdoor activities affected by weather
- Travel affected by weather
- Bike rides affected by weather
- Walking affected by weather
- Farming questions involving weather
- What to wear based on weather
- Weather comparisons
- Historical weather
- Future weather

If an activity depends on weather, classify it as weather.

For weather questions determine whether it is:

forecast

OR

weather_knowledge

FORECAST means the user needs actual weather data.

WEATHER_KNOWLEDGE means the user wants an explanation
about weather itself.

If the question is unrelated to weather, classify it as
non_weather.

Return ONLY JSON.

Format:

{{
    "scope": "weather" or "non_weather",
    "query_type": "forecast" or "weather_knowledge",
    "intent": "SHORT_INTENT_NAME",
    "requested_city": "CITY_NAME or null"
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

                parsed = json.loads(
                    cleaned
                )

                scope = parsed.get(
                    "scope",
                    "non_weather"
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
                    scope = "non_weather"

                if query_type not in [
                    "forecast",
                    "weather_knowledge"
                ]:
                    query_type = "forecast"

                return {
                    "scope": scope,
                    "query_type": query_type,
                    "intent": str(intent).upper(),
                    "requested_city": requested_city,
                    "is_weather_related": scope == "weather"
                }

            except Exception as error:

                print(
                    "Classifier JSON error:",
                    str(error)
                )

    # ========================================================
    # STEP 3 — FINAL LOCAL FALLBACK
    # ========================================================

    is_weather = looks_weather_related(
        message
    )

    if not is_weather:

        return {
            "scope": "non_weather",
            "query_type": "weather_knowledge",
            "intent": "OUT_OF_SCOPE",
            "requested_city": None,
            "is_weather_related": False
        }

    lower = message.lower()

    forecast_words = [
        "today",
        "tomorrow",
        "forecast",
        "current",
        "now",
        "will",
        "next",
        "this week",
        "this weekend",
        "day after",
        "temperature today",
        "temperature tomorrow",
        "weather today",
        "weather tomorrow"
    ]

    query_type = (
        "forecast"
        if any(
            word in lower
            for word in forecast_words
        )
        else "weather_knowledge"
    )

    return {
        "scope": "weather",
        "query_type": query_type,
        "intent": "GENERAL_WEATHER",
        "requested_city": extract_requested_city(message),
        "is_weather_related": True
    }


# ============================================================
# DATE HELPERS
# ============================================================

def get_previous_target_date(
    history
):

    for item in reversed(history):

        target = item.get(
            "targetDate"
        )

        if target:

            try:

                return date.fromisoformat(
                    target
                )

            except:
                pass

    return None


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

    for name, number in weekdays.items():

        if re.search(
            rf"\b{name}\b",
            text
        ):

            days_ahead = (
                number
                - reference_date.weekday()
            ) % 7

            return days_ahead

    return None


def get_forecast_index(
    message: str,
    available_dates: list[str],
    history
):

    text = message.lower().strip()

    today = date.today()

    # --------------------------------------------------------
    # TODAY
    # --------------------------------------------------------

    if re.search(
        r"\btoday\b",
        text
    ):

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
        re.search(
            r"\btomorrow\b",
            text
        )

        or re.search(
            r"\btmrw\b|\btmr\b",
            text
        )
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
        r"\bin\s+"
        r"(\d+|one|two|three|four|five|six|seven)"
        r"\s+days?\b",
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
    # FOLLOW-UP QUESTIONS
    # --------------------------------------------------------

    previous_target = (
        get_previous_target_date(
            history
        )
    )

    if previous_target:

        difference = (
            previous_target - today
        ).days

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

            return max(
                difference,
                0
            )

        if any(
            text.startswith(prefix)

            for prefix in [

                "what about",

                "how about",

                "and "
            ]
        ):

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
    # Only forecast questions reach this node.
    # If no date is specified, tomorrow remains the
    # existing project behavior.
    # --------------------------------------------------------

    return 1


# ============================================================
# GEOCODING
# ============================================================

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


# ============================================================
# WEATHER API
# ============================================================

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

    async with httpx.AsyncClient(
        timeout=10
    ) as client:

        response = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params=params
        )

        response.raise_for_status()

        return response.json()


# ============================================================
# WEATHER DATA NODE
# ============================================================

async def weather_node(
    state: State
):

    city = (

        state.get(
            "requested_city"
        )

        or state.get(
            "city"
        )

        or "Nashik"
    )

    place = await geocode(
        city
    )

    data = await weather_for(

        place["latitude"],

        place["longitude"]
    )

    available_dates = (
        data["daily"]["time"]
    )

    requested_index = (
        get_forecast_index(

            state["message"],

            available_dates,

            state.get(
                "history",
                []
            )
        )
    )

    forecast_index = min(

        max(
            requested_index,
            0
        ),

        len(
            available_dates
        ) - 1
    )

    target_date = (
        available_dates[
            forecast_index
        ]
    )

    location = {

        "name":
            place["name"],

        "country":
            place.get("country"),

        "latitude":
            place["latitude"],

        "longitude":
            place["longitude"]
    }

    return {

        "location":
            location,

        "weather":
            data,

        "target_date":
            target_date,

        "forecast_index":
            forecast_index
    }


# ============================================================
# HISTORICAL WEATHER FOR V4 PREVIOUS-DAY FEATURES
# ============================================================

async def historical_weather_for(
    lat: float,
    lon: float,
    target_date: str
):
    """
    Fetch the previous day's observed/reanalysis weather.

    V4 was trained as a next-day model:
    day T features -> day T+1 risk.

    Therefore, when predicting tomorrow, the model uses
    today's weather as the main features and yesterday's
    weather as the previous-day features.
    """

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": target_date,
        "end_date": target_date,
        "daily": (
            "temperature_2m_max,"
            "relative_humidity_2m_mean,"
            "precipitation_sum,"
            "wind_speed_10m_max,"
            "surface_pressure_mean"
        ),
        "timezone": "auto"
    }

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params=params
        )

        response.raise_for_status()

        data = response.json()

    daily = data.get("daily", {})

    if not daily.get("time"):
        raise ValueError(
            f"No historical weather data available for {target_date}"
        )

    def first_value(key: str, default: float = 0.0):
        values = daily.get(key, [])
        if not values or values[0] is None:
            return default
        return float(values[0])

    return {
        "temperature": first_value("temperature_2m_max"),
        "humidity": first_value("relative_humidity_2m_mean"),
        "rainfall": first_value("precipitation_sum"),
        "wind_speed": first_value("wind_speed_10m_max"),
        "pressure": first_value("surface_pressure_mean", 1013.0)
    }


# ============================================================
# ML RISK ANALYSIS
# ============================================================

async def risk_node(
    state: State
):

    weather = state["weather"]
    daily = weather["daily"]
    index = state["forecast_index"]

    # --------------------------------------------------------
    # TARGET-DAY WEATHER
    # These values are used for the user-facing answer.
    # --------------------------------------------------------

    rainfall = float(
        daily["precipitation_sum"][index] or 0
    )

    rain_probability = float(
        daily["precipitation_probability_max"][index] or 0
    )

    wind = float(
        daily["wind_speed_10m_max"][index] or 0
    )

    temperature = float(
        daily["temperature_2m_max"][index] or 0
    )

    current = weather.get("current", {})

    humidity = float(
        current.get(
            "relative_humidity_2m",
            0
        ) or 0
    )

    pressure = float(
        current.get(
            "surface_pressure",
            1013
        ) or 1013
    )

    features = {
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": rainfall,
        "rain_probability": rain_probability,
        "wind_speed": wind,
        "pressure": pressure
    }

    # --------------------------------------------------------
    # V4 MODEL FEATURES
    #
    # V4 is a NEXT-DAY model trained on Nashik:
    #
    # day T features + previous day features
    #                 -> day T+1 risk
    #
    # Therefore we only use V4 for:
    #   1. Nashik
    #   2. Tomorrow (forecast index 1)
    #
    # For today or dates farther than tomorrow, the V4 model
    # would not match the training target, so we use the
    # transparent rule-based fallback instead.
    # --------------------------------------------------------

    location_name = (
        state.get("location", {}).get("name", "")
        or ""
    )

    is_nashik = (
        location_name.strip().lower() == "nashik"
    )

    can_use_v4 = (
        is_nashik
        and index == 1
    )

    ml_features = {
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": rainfall,
        "wind_speed": wind,
        "pressure": pressure,
        "previous_rainfall": 0.0,
        "previous_wind_speed": 0.0,
        "previous_humidity": 0.0,
        "month": int(
            date.fromisoformat(
                state["target_date"]
            ).month
        ),
        "day_of_year": int(
            date.fromisoformat(
                state["target_date"]
            ).timetuple().tm_yday
        )
    }

    risk = None

    # --------------------------------------------------------
    # V4 NEXT-DAY PREDICTION
    # --------------------------------------------------------

    if can_use_v4:

        try:

            target = date.fromisoformat(
                state["target_date"]
            )

            previous_date = (
                target - timedelta(days=2)
            )

            # Example:
            # target = tomorrow
            # model input day T = today
            # previous day = yesterday
            #
            # Since target is T+1, the previous feature date
            # is target - 2 days.
            previous = await historical_weather_for(
                state["location"]["latitude"],
                state["location"]["longitude"],
                previous_date.isoformat()
            )

            # For the day T features, use today's forecast/current
            # weather because the model predicts T+1.
            day_t_index = 0

            day_t_rainfall = float(
                daily["precipitation_sum"][day_t_index] or 0
            )

            day_t_wind = float(
                daily["wind_speed_10m_max"][day_t_index] or 0
            )

            day_t_temperature = float(
                daily["temperature_2m_max"][day_t_index] or 0
            )

            day_t_humidity = float(
                current.get(
                    "relative_humidity_2m",
                    0
                ) or 0
            )

            day_t_pressure = float(
                current.get(
                    "surface_pressure",
                    1013
                ) or 1013
            )

            ml_features = {
                "temperature": day_t_temperature,
                "humidity": day_t_humidity,
                "rainfall": day_t_rainfall,
                "wind_speed": day_t_wind,
                "pressure": day_t_pressure,
                "previous_rainfall": previous["rainfall"],
                "previous_wind_speed": previous["wind_speed"],
                "previous_humidity": previous["humidity"],
                "month": int(target.month),
                "day_of_year": int(
                    target.timetuple().tm_yday
                )
            }

            async with httpx.AsyncClient(
                timeout=10
            ) as client:

                response = await client.post(
                    f"{ML_SERVICE_URL}/predict",
                    json={
                        **ml_features,
                        "rain_probability": rain_probability
                    }
                )

                response.raise_for_status()

                risk = response.json()

                print(
                    "V4 ML RESULT:",
                    risk
                )

            risk["source"] = "V4_next_day_Nashik"

        except Exception as error:

            print(
                "V4 ML service error:",
                str(error)
            )
            print("Using rule-based fallback.")

            risk = None

    # --------------------------------------------------------
    # RULE-BASED FALLBACK
    # --------------------------------------------------------

    if risk is None:

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
            "risk_score": None,
            "confidence": None,
            "probabilities": {},
            "source": "rule_based"
        }

    alert = None

    if risk.get("risk_level") == "High":

        alert = (
            f"High weather risk detected for "
            f"{state['location']['name']} "
            f"on {state['target_date']}. "
            f"Expected rainfall is about "
            f"{rainfall:.0f} mm with a "
            f"{rain_probability:.0f}% "
            f"rain probability. "
            f"Check updated local weather "
            f"advisories."
        )

    return {
        "features": features,
        "ml_features": ml_features,
        "risk": risk,
        "alert": alert
    }


# ============================================================
# NON-WEATHER RESPONSE
# ============================================================

def out_of_scope_response():

    return (

        "I'm WeatherGPT, and this website is "
        "designed specifically for weather-related "
        "information. I can help with forecasts, "
        "rain, temperature, wind, storms, climate, "
        "weather science, weather safety, travel "
        "conditions and other weather-related "
        "questions."
    )


# ============================================================
# WEATHER KNOWLEDGE RESPONSE
# ============================================================

async def weather_knowledge_response(
    state: State
):

    prompt = f"""
You are WeatherGPT.

You are a specialized weather assistant.

Answer the user's weather-related question.

This is a WEATHER KNOWLEDGE question, not necessarily
a forecast question.

USER QUESTION:
{state["message"]}

RECENT CONVERSATION:
{state.get("history", [])}

RULES:

1. Answer the actual question directly.

2. Stay within weather, meteorology, climate,
   atmospheric science and weather-related safety.

3. Do NOT automatically provide today's weather.

4. Do NOT automatically provide tomorrow's weather.

5. Do NOT invent live weather information.

6. If the user asks a conceptual question, explain
   the concept clearly.

7. If useful, give a simple real-world example.

8. If the user asks a weather-related "why" question,
   explain the reason.

9. Keep the answer easy to understand.

10. Do not say that you are a general-purpose assistant.

11. Keep the answer around 3-6 sentences.
"""

    result = await call_gemini(
        prompt
    )

    if result:

        return result

    return (
        "I can explain that weather-related topic, "
        "but the AI explanation service is currently "
        "unavailable."
    )


# ============================================================
# FORECAST RESPONSE
# ============================================================

async def forecast_response(
    state: State
):

    location = (
        state["location"]["name"]
    )

    target_date = (
        state["target_date"]
    )

    features = (
        state["features"]
    )

    risk_level = (
        state["risk"].get(
            "risk_level",
            "Low"
        )
    )

    rainfall = (
        features["rainfall"]
    )

    rain_probability = (
        features["rain_probability"]
    )

    temperature = (
        features["temperature"]
    )

    wind = (
        features["wind_speed"]
    )

    humidity = (
        features["humidity"]
    )

    pressure = (
        features["pressure"]
    )

    prompt = f"""
You are WeatherGPT, a conversational weather assistant.

Answer the user's ACTUAL question.

USER QUESTION:
{state["message"]}

LOCATION:
{location}

FORECAST DATE:
{target_date}

WEATHER DATA:

Temperature:
{temperature:.1f} °C

Rain probability:
{rain_probability:.0f} %

Expected rainfall:
{rainfall:.1f} mm

Wind speed:
{wind:.1f} km/h

Humidity:
{humidity:.0f} %

Pressure:
{pressure:.1f} hPa

ML WEATHER RISK:
{risk_level}

RECENT CONVERSATION:
{state.get("history", [])}

RULES:

1. Answer the user's actual question.

2. Use ONLY the supplied weather values.

3. Never invent weather values.

4. Never automatically answer with today's weather.

5. Never automatically answer with tomorrow's weather.

6. Use the correct forecast date.

7. For rain questions, focus on rain probability
   and expected rainfall.

8. For temperature questions, focus on temperature.

9. For wind questions, focus on wind.

10. For travel, cycling, walking or outdoor questions,
    explain how the weather conditions affect the activity.

11. For comparisons, clearly compare the requested dates
    only if the required data is available.

12. For weather safety questions, use the weather values
    and ML risk as supporting information.

13. Do not claim the ML model is a professional
    meteorological warning system.

14. If the risk is High, tell the user to check
    current local weather advisories.

15. Keep the response natural.

16. Usually answer in 2-5 sentences.

17. Do not repeat the same generic response for every
    question.
"""

    result = await call_gemini(
        prompt
    )

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

            f"For {target_date} in "
            f"{location}, the rain probability "
            f"is around "
            f"{rain_probability:.0f}% with "
            f"expected rainfall of about "
            f"{rainfall:.0f} mm."
        )

    if intent == "TEMPERATURE":

        return (

            f"On {target_date} in "
            f"{location}, the forecast high "
            f"is around "
            f"{temperature:.1f}°C."
        )

    if intent == "WIND":

        return (

            f"On {target_date} in "
            f"{location}, maximum wind speed "
            f"is expected to be around "
            f"{wind:.1f} km/h."
        )

    return (

        f"For {target_date} in "
        f"{location}, the weather risk is "
        f"{risk_level.lower()}. "
        f"Rain probability is around "
        f"{rain_probability:.0f}%."
    )


# ============================================================
# RESPONSE NODE
# ============================================================

async def response_node(
    state: State
):

    scope = state.get(
        "scope",
        "non_weather"
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

            "response":
                out_of_scope_response(),

            "is_weather_related":
                False
        }

    # --------------------------------------------------------
    # WEATHER KNOWLEDGE
    # --------------------------------------------------------

    if query_type == "weather_knowledge":

        response = (
            await weather_knowledge_response(
                state
            )
        )

        return {

            "response":
                response,

            "is_weather_related":
                True
        }

    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    response = await forecast_response(
        state
    )

    return {

        "response":
            response,

        "is_weather_related":
            True
    }


# ============================================================
# LANGGRAPH
# ============================================================

workflow = StateGraph(
    State
)

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


# ============================================================
# CONDITIONAL ROUTING
# ============================================================

def route_after_query(
    state: State
):

    # Non-weather:
    # go directly to response.
    if state.get(
        "scope"
    ) == "non_weather":

        return "response_generation"

    # Weather knowledge:
    # no need for forecast or ML.
    if state.get(
        "query_type"
    ) == "weather_knowledge":

        return "response_generation"

    # Forecast:
    # fetch weather and use ML.
    return "weather_data"


workflow.add_conditional_edges(

    "query_understanding",

    route_after_query,

    {

        "weather_data":
            "weather_data",

        "response_generation":
            "response_generation"
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
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {

        "ok": True,

        "langgraph": True,

        "gemini_configured":
            bool(GEMINI_API_KEY),

        "model":
            GEMINI_MODEL
    }


# ============================================================
# CHAT
# ============================================================

@app.post("/chat")
async def chat(
    req: ChatRequest
):

    result = await graph.ainvoke(

        {

            "message":
                req.message,

            "city":
                req.city,

            "history":
                req.history
        }
    )

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

        "ml_features":
            result.get(
                "ml_features"
            ),

        "is_weather_related":
            result.get(
                "is_weather_related",
                False
            ),

        "query_type":
            result.get(
                "query_type"
            ),

        "timestamp":
            datetime.utcnow().isoformat()
            + "Z"
    }