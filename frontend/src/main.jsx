import React, { useEffect, useMemo, useState } from "react";

import { createRoot } from "react-dom/client";

import {
  MapPin,
  Send,
  Thermometer,
  Droplets,
  Wind,
  AlertTriangle,
  CloudRain,
  Gauge,
  CalendarDays,
  Umbrella,
  RefreshCw,
  Sun,
  Cloud,
  CloudSun,
  CloudLightning,
  Navigation,
  Sparkles,
  MessageCircle,
  History,
  Activity,
  ArrowUpRight,
} from "lucide-react";

import "./styles.css";

// ============================================================
// CONFIG
// ============================================================

const API = "http://localhost:4000";

const SESSION_STORAGE_KEY = "weathergpt_session_id";

// ============================================================
// SESSION HELPERS
// ============================================================

function createSessionId() {
  return `session-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function getOrCreateSessionId() {
  const existing = localStorage.getItem(SESSION_STORAGE_KEY);

  if (existing) {
    return existing;
  }

  const newSessionId = createSessionId();

  localStorage.setItem(SESSION_STORAGE_KEY, newSessionId);

  return newSessionId;
}

function createNewSession() {
  const newSessionId = createSessionId();

  localStorage.setItem(SESSION_STORAGE_KEY, newSessionId);

  return newSessionId;
}

// ============================================================
// WEATHER ICON
// ============================================================

function getWeatherIcon(code, size = 24) {
  if (code === undefined || code === null) {
    return <CloudSun size={size} />;
  }

  if (code === 0) {
    return <Sun size={size} />;
  }

  if ([1, 2].includes(code)) {
    return <CloudSun size={size} />;
  }

  if (code === 3) {
    return <Cloud size={size} />;
  }

  if ([45, 48].includes(code)) {
    return <Cloud size={size} />;
  }

  if ([51, 53, 55, 56, 57].includes(code)) {
    return <CloudRain size={size} />;
  }

  if ([61, 63, 65, 66, 67, 80, 81, 82].includes(code)) {
    return <CloudRain size={size} />;
  }

  if ([95, 96, 99].includes(code)) {
    return <CloudLightning size={size} />;
  }

  return <CloudSun size={size} />;
}

// ============================================================
// DATE HELPERS
// ============================================================

function formatDate(dateString) {
  if (!dateString) {
    return "";
  }

  const date = new Date(`${dateString}T00:00:00`);

  return date.toLocaleDateString("en-IN", {
    weekday: "long",
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

function formatShortDate(dateString) {
  if (!dateString) {
    return "";
  }

  const date = new Date(`${dateString}T00:00:00`);

  return date.toLocaleDateString("en-IN", {
    weekday: "short",
    day: "numeric",
    month: "short",
  });
}

// ============================================================
// RISK
// ============================================================

function riskClass(level) {
  if (!level) {
    return "risk-neutral";
  }

  return `risk-${String(level).toLowerCase()}`;
}

// ============================================================
// APP
// ============================================================

function App() {
  // ----------------------------------------------------------
  // CITY
  // ----------------------------------------------------------

  const [city, setCity] = useState("Nashik");

  const [cityInput, setCityInput] = useState("Nashik");

  // ----------------------------------------------------------
  // WEATHER
  // ----------------------------------------------------------

  const [forecast, setForecast] = useState(null);

  const [loadingWeather, setLoadingWeather] = useState(false);

  // ----------------------------------------------------------
  // CHAT
  // ----------------------------------------------------------

  const [question, setQuestion] = useState("");

  const [messages, setMessages] = useState([]);

  const [loadingChat, setLoadingChat] = useState(false);

  // ----------------------------------------------------------
  // SESSION
  // ----------------------------------------------------------

  const [sessionId, setSessionId] = useState(null);

  const [loadingHistory, setLoadingHistory] = useState(false);

  // ----------------------------------------------------------
  // UI
  // ----------------------------------------------------------

  const [error, setError] = useState("");

  const [selectedDetails, setSelectedDetails] = useState(null);

  // ==========================================================
  // LOAD WEATHER
  // ==========================================================

  async function loadWeather(targetCity = city) {
    setLoadingWeather(true);

    setError("");

    try {
      const response = await fetch(
        `${API}/api/weather/forecast?city=${encodeURIComponent(targetCity)}`,
      );

      if (!response.ok) {
        throw new Error("Unable to fetch weather data.");
      }

      const data = await response.json();

      setForecast(data);
    } catch (err) {
      console.error(err);

      setError(err.message || "Weather service unavailable.");
    } finally {
      setLoadingWeather(false);
    }
  }

  // ==========================================================
  // LOAD CHAT HISTORY
  // ==========================================================

  async function loadConversationHistory(currentSessionId) {
    if (!currentSessionId) {
      return;
    }

    setLoadingHistory(true);

    try {
      const response = await fetch(
        `${API}/api/chat/history/${encodeURIComponent(currentSessionId)}`,
      );

      if (!response.ok) {
        throw new Error(`History request failed (${response.status})`);
      }

      const data = await response.json();

      const history = Array.isArray(data)
        ? data
        : Array.isArray(data?.messages)
          ? data.messages
          : [];

      const restoredMessages = history.map((message, index) => ({
        role: message.role,

        content: message.content,

        timestamp:
          message.createdAt ||
          new Date(Date.now() - (history.length - index) * 1000).toISOString(),

        targetDate: message.targetDate ?? message.target_date ?? null,

        forecastIndex: message.forecastIndex ?? message.forecast_index ?? null,

        features: message.features ?? null,

        risk:
          message.risk ??
          (message.riskLevel
            ? {
                risk_level: message.riskLevel,
              }
            : null),

        intent: message.intent ?? null,
      }));

      setMessages(restoredMessages);

      const latestAssistant = [...restoredMessages]
        .reverse()
        .find((message) => message.role === "assistant");

      if (latestAssistant) {
        setSelectedDetails(latestAssistant);
      }
    } catch (err) {
      console.error("History loading error:", err);

      setMessages([]);
    } finally {
      setLoadingHistory(false);
    }
  }

  // ==========================================================
  // INITIALIZE
  // ==========================================================

  useEffect(() => {
    async function initialize() {
      const savedSessionId = getOrCreateSessionId();

      setSessionId(savedSessionId);

      await Promise.all([
        loadWeather("Nashik"),
        loadConversationHistory(savedSessionId),
      ]);
    }

    initialize();
  }, []);

  // ==========================================================
  // CURRENT WEATHER
  // ==========================================================

  const current = forecast?.current;

  const currentTemp = current?.temperature_2m;

  const currentHumidity = current?.relative_humidity_2m;

  const currentWind = current?.wind_speed_10m;

  const currentPressure = current?.surface_pressure;

  // ==========================================================
  // 7-DAY FORECAST
  // ==========================================================

  const forecastDays = useMemo(() => {
    if (!forecast?.daily?.time) {
      return [];
    }

    return forecast.daily.time.map((date, index) => ({
      date,

      max: forecast.daily.temperature_2m_max?.[index],

      min: forecast.daily.temperature_2m_min?.[index],

      rain: forecast.daily.precipitation_probability_max?.[index],

      rainfall: forecast.daily.precipitation_sum?.[index],

      wind: forecast.daily.wind_speed_10m_max?.[index],

      code: forecast.daily.weather_code?.[index],
    }));
  }, [forecast]);

  // ==========================================================
  // UPDATE CITY
  // ==========================================================

  async function handleCityUpdate() {
    const newCity = cityInput.trim();

    if (!newCity) {
      return;
    }

    setCity(newCity);

    const newSessionId = createNewSession();

    setSessionId(newSessionId);

    setMessages([]);

    setSelectedDetails(null);

    setError("");

    await loadWeather(newCity);
  }

  // ==========================================================
  // CHAT
  // ==========================================================

  async function askWeatherGPT(event) {
    event?.preventDefault();

    const text = question.trim();

    if (!text || loadingChat) {
      return;
    }

    setError("");

    setQuestion("");

    setLoadingChat(true);

    let currentSessionId = sessionId;

    if (!currentSessionId) {
      currentSessionId = getOrCreateSessionId();

      setSessionId(currentSessionId);
    }

    const userMessage = {
      role: "user",

      content: text,

      timestamp: new Date().toISOString(),
    };

    setMessages((previous) => [...previous, userMessage]);

    try {
      const history = messages.map((message) => ({
        role: message.role,

        content: message.content,

        targetDate: message.targetDate ?? null,

        forecastIndex: message.forecastIndex ?? null,

        features: message.features ?? null,

        risk: message.risk ?? null,

        intent: message.intent ?? null,
      }));

      const response = await fetch(`${API}/api/chat`, {
        method: "POST",

        headers: {
          "Content-Type": "application/json",
        },

        body: JSON.stringify({
          message: text,

          city,

          history,

          sessionId: currentSessionId,
        }),
      });

      if (!response.ok) {
        let detail = "";

        try {
          const errorData = await response.json();

          detail = errorData?.detail
            ? JSON.stringify(errorData.detail)
            : errorData?.error || "";
        } catch {
          // Ignore JSON errors.
        }

        throw new Error(
          `Request failed (${response.status})${detail ? `: ${detail}` : ""}`,
        );
      }

      const data = await response.json();

      console.log("WeatherGPT response:", data);

      if (data.sessionId) {
        localStorage.setItem(SESSION_STORAGE_KEY, data.sessionId);

        setSessionId(data.sessionId);
      }

      const assistantMessage = {
        role: "assistant",

        content: data.response || "I couldn't generate a response.",

        timestamp: new Date().toISOString(),

        targetDate: data.target_date,

        forecastIndex: data.forecast_index,

        features: data.features,

        risk: data.risk,

        intent: data.intent,
      };

      setMessages((previous) => [...previous, assistantMessage]);

      setSelectedDetails(assistantMessage);

      await loadWeather(city);
    } catch (err) {
      console.error("Chat error:", err);

      setError(err.message || "Unable to contact WeatherGPT.");

      setMessages((previous) =>
        previous.filter((message) => message !== userMessage),
      );
    } finally {
      setLoadingChat(false);
    }
  }

  // ==========================================================
  // CLEAR CHAT
  // ==========================================================

  function clearChat() {
    const newSessionId = createNewSession();

    setSessionId(newSessionId);

    setMessages([]);

    setSelectedDetails(null);

    setError("");
  }

  // ==========================================================
  // SELECT FORECAST
  // ==========================================================

  function selectForecastDay(day, index) {
    setSelectedDetails({
      role: "assistant",

      targetDate: day.date,

      forecastIndex: index,

      features: {
        temperature: day.max,

        humidity: currentHumidity,

        rainfall: day.rainfall,

        rain_probability: day.rain,

        wind_speed: day.wind,

        pressure: currentPressure,
      },

      risk: selectedDetails?.risk || null,

      intent: "FORECAST",
    });
  }

  // ==========================================================
  // RENDER
  // ==========================================================

  return (
    <div className="app">
      {/* ====================================================
          TOP NAVIGATION
      ==================================================== */}

      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <CloudSun size={25} strokeWidth={2.2} />
          </div>

          <div className="brand-copy">
            <h1>WeatherGPT</h1>

            <span>AI weather intelligence</span>
          </div>
        </div>

        <div className="topbar-right">
          <div className="ai-status">
            <span className="status-dot"></span>
            AI Agent Online
          </div>

          <div className="location-box">
            <MapPin size={16} />

            <input
              value={cityInput}
              onChange={(event) => setCityInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  handleCityUpdate();
                }
              }}
              placeholder="Enter city"
            />
          </div>

          <button
            className="location-button"
            onClick={handleCityUpdate}
            disabled={loadingWeather}
          >
            {loadingWeather ? (
              <RefreshCw size={16} className="spin" />
            ) : (
              <Navigation size={16} />
            )}
            Update
          </button>
        </div>
      </header>

      {/* ====================================================
          MAIN
      ==================================================== */}

      <main className="main">
        {/* ==================================================
            HERO
        ================================================== */}

        <section className="hero">
          <div className="hero-content">
            <div className="eyebrow">
              <Sparkles size={15} />
              INTELLIGENT WEATHER ASSISTANT
            </div>

            <h2>
              Weather data,
              <span> understood.</span>
            </h2>

            <p>
              Ask questions naturally. WeatherGPT combines live forecast data,
              machine learning and AI reasoning to help you understand what the
              weather means.
            </p>

            <div className="hero-pills">
              <span>
                <Activity size={14} />
                Live forecast
              </span>

              <span>
                <Sparkles size={14} />
                AI reasoning
              </span>

              <span>
                <AlertTriangle size={14} />
                ML risk analysis
              </span>
            </div>
          </div>

          <div className="hero-visual">
            <div className="hero-orb orb-one"></div>

            <div className="hero-orb orb-two"></div>

            <div className="hero-weather-icon">
              {getWeatherIcon(forecast?.daily?.weather_code?.[0], 58)}
            </div>
          </div>
        </section>

        {/* ==================================================
            ERROR
        ================================================== */}

        {error && (
          <div className="error-banner">
            <AlertTriangle size={18} />

            <span>{error}</span>
          </div>
        )}

        {/* ==================================================
            OVERVIEW
        ================================================== */}

        <section className="overview-grid">
          {/* Temperature */}

          <div className="stat-card">
            <div className="stat-icon blue">
              <Thermometer size={20} />
            </div>

            <div className="stat-content">
              <span>Temperature</span>

              <strong>
                {currentTemp !== undefined
                  ? `${currentTemp.toFixed(1)}°`
                  : "--"}
              </strong>

              <small>Current</small>
            </div>
          </div>

          {/* Humidity */}

          <div className="stat-card">
            <div className="stat-icon cyan">
              <Droplets size={20} />
            </div>

            <div className="stat-content">
              <span>Humidity</span>

              <strong>
                {currentHumidity !== undefined ? `${currentHumidity}%` : "--"}
              </strong>

              <small>Relative humidity</small>
            </div>
          </div>

          {/* Wind */}

          <div className="stat-card">
            <div className="stat-icon purple">
              <Wind size={20} />
            </div>

            <div className="stat-content">
              <span>Wind</span>

              <strong>
                {currentWind !== undefined ? `${currentWind.toFixed(1)}` : "--"}

                <em>km/h</em>
              </strong>

              <small>Current wind</small>
            </div>
          </div>

          {/* Pressure */}

          <div className="stat-card">
            <div className="stat-icon orange">
              <Gauge size={20} />
            </div>

            <div className="stat-content">
              <span>Pressure</span>

              <strong>
                {currentPressure !== undefined
                  ? `${currentPressure.toFixed(0)}`
                  : "--"}

                <em>hPa</em>
              </strong>

              <small>Atmospheric pressure</small>
            </div>
          </div>
        </section>

        {/* ==================================================
            MAIN GRID
        ================================================== */}

        <section className="dashboard-grid">
          {/* =================================================
              CHAT PANEL
          ================================================= */}

          <section className="chat-panel panel">
            <div className="panel-top">
              <div className="panel-title">
                <div className="panel-icon chat-icon">
                  <MessageCircle size={18} />
                </div>

                <div>
                  <h3>Ask WeatherGPT</h3>

                  <p>Conversational weather intelligence</p>
                </div>
              </div>

              {messages.length > 0 && (
                <button className="clear-button" onClick={clearChat}>
                  <RefreshCw size={14} />
                  New chat
                </button>
              )}
            </div>

            {/* CHAT */}

            <div className="messages">
              {messages.length === 0 && !loadingHistory && (
                <div className="empty-chat">
                  <div className="empty-icon">
                    <Sparkles size={28} />
                  </div>

                  <h4>Ask anything about the weather</h4>

                  <p>
                    I can analyze rain, temperature, wind, travel conditions and
                    upcoming weather.
                  </p>

                  <div className="suggestion-list">
                    <button
                      onClick={() => setQuestion("Will it rain tomorrow?")}
                    >
                      Will it rain tomorrow?
                    </button>

                    <button
                      onClick={() => setQuestion("Should I go out on Monday?")}
                    >
                      Should I go out Monday?
                    </button>

                    <button
                      onClick={() =>
                        setQuestion("How windy will it be tomorrow?")
                      }
                    >
                      How windy tomorrow?
                    </button>
                  </div>
                </div>
              )}

              {loadingHistory && (
                <div className="history-loading">
                  <RefreshCw size={18} className="spin" />
                  Restoring conversation...
                </div>
              )}

              {messages.map((message, index) => (
                <div
                  key={`${message.timestamp}-${index}`}
                  className={
                    message.role === "user"
                      ? "message-row user"
                      : "message-row assistant"
                  }
                >
                  {message.role === "assistant" && (
                    <div className="avatar ai-avatar">
                      <Sparkles size={15} />
                    </div>
                  )}

                  <div className="message-wrapper">
                    <div className="message-name">
                      {message.role === "user" ? "You" : "WeatherGPT"}
                    </div>

                    <div className="message-bubble">{message.content}</div>

                    {message.role === "assistant" && message.targetDate && (
                      <button
                        className="message-context"
                        onClick={() => setSelectedDetails(message)}
                      >
                        <CalendarDays size={13} />

                        {formatShortDate(message.targetDate)}

                        <ArrowUpRight size={12} />
                      </button>
                    )}
                  </div>

                  {message.role === "user" && (
                    <div className="avatar user-avatar">You</div>
                  )}
                </div>
              ))}

              {loadingChat && (
                <div className="message-row assistant">
                  <div className="avatar ai-avatar">
                    <Sparkles size={15} />
                  </div>

                  <div className="message-wrapper">
                    <div className="message-name">WeatherGPT</div>

                    <div className="message-bubble typing-bubble">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* CHAT INPUT */}

            <form className="chat-form" onSubmit={askWeatherGPT}>
              <div className="input-shell">
                <MessageCircle size={18} />

                <input
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder="Ask about rain, travel, temperature..."
                  disabled={loadingChat}
                />
              </div>

              <button
                type="submit"
                className="send-button"
                disabled={loadingChat || !question.trim()}
              >
                {loadingChat ? (
                  <RefreshCw size={19} className="spin" />
                ) : (
                  <Send size={19} />
                )}
              </button>
            </form>
          </section>

          {/* =================================================
              FORECAST PANEL
          ================================================= */}

          <section className="forecast-panel panel">
            <div className="panel-top">
              <div className="panel-title">
                <div className="panel-icon forecast-icon">
                  <CalendarDays size={18} />
                </div>

                <div>
                  <h3>7-day forecast</h3>

                  <p>{forecast?.location?.name || city}</p>
                </div>
              </div>

              <span className="forecast-live">
                <span></span>
                Live
              </span>
            </div>

            <div className="forecast-list">
              {forecastDays.map((day, index) => (
                <button
                  key={day.date}
                  className={
                    selectedDetails?.targetDate === day.date
                      ? "forecast-row active"
                      : "forecast-row"
                  }
                  onClick={() => selectForecastDay(day, index)}
                >
                  <div className="forecast-date">
                    <strong>{formatShortDate(day.date)}</strong>

                    {index === 0 && <span>Today</span>}
                  </div>

                  <div className="forecast-weather-icon">
                    {getWeatherIcon(day.code, 24)}
                  </div>

                  <div className="forecast-rain">
                    <Umbrella size={13} />
                    {day.rain ?? 0}%
                  </div>

                  <div className="forecast-temperature">
                    <strong>{day.max?.toFixed(1)}°</strong>

                    <span>/{day.min?.toFixed(1)}°</span>
                  </div>
                </button>
              ))}
            </div>
          </section>
        </section>

        {/* ==================================================
            AI ANALYSIS
        ================================================== */}

        {selectedDetails && (
          <section className="analysis-panel panel">
            <div className="analysis-heading">
              <div>
                <div className="analysis-label">
                  <Sparkles size={14} />
                  WEATHERGPT ANALYSIS
                </div>

                <h3>
                  {selectedDetails.targetDate
                    ? formatDate(selectedDetails.targetDate)
                    : "Forecast analysis"}
                </h3>
              </div>

              {selectedDetails.risk?.risk_level && (
                <span
                  className={`risk-badge ${riskClass(
                    selectedDetails.risk.risk_level,
                  )}`}
                >
                  <span></span>

                  {selectedDetails.risk.risk_level}
                </span>
              )}
            </div>

            {selectedDetails.features && (
              <div className="analysis-grid">
                <div className="analysis-card">
                  <Thermometer size={18} />

                  <span>Temperature</span>

                  <strong>
                    {selectedDetails.features.temperature?.toFixed(1)}°C
                  </strong>
                </div>

                <div className="analysis-card">
                  <Droplets size={18} />

                  <span>Humidity</span>

                  <strong>{selectedDetails.features.humidity}%</strong>
                </div>

                <div className="analysis-card">
                  <Umbrella size={18} />

                  <span>Rain probability</span>

                  <strong>{selectedDetails.features.rain_probability}%</strong>
                </div>

                <div className="analysis-card">
                  <CloudRain size={18} />

                  <span>Expected rainfall</span>

                  <strong>{selectedDetails.features.rainfall} mm</strong>
                </div>

                <div className="analysis-card">
                  <Wind size={18} />

                  <span>Maximum wind</span>

                  <strong>
                    {selectedDetails.features.wind_speed?.toFixed(1)} km/h
                  </strong>
                </div>

                <div className="analysis-card">
                  <Gauge size={18} />

                  <span>Pressure</span>

                  <strong>
                    {selectedDetails.features.pressure?.toFixed(1)} hPa
                  </strong>
                </div>
              </div>
            )}

            <div className="analysis-footer">
              <span>
                <CalendarDays size={14} />
                Forecast:{" "}
                {selectedDetails.targetDate
                  ? formatDate(selectedDetails.targetDate)
                  : "--"}
              </span>

              {selectedDetails.intent && (
                <span>
                  Intent: {selectedDetails.intent.replaceAll("_", " ")}
                </span>
              )}
            </div>
          </section>
        )}
      </main>

      {/* ====================================================
          FOOTER
      ==================================================== */}

      <footer className="footer">
        <div>
          WeatherGPT
          <span>•</span>
          SIH 2026
          <span>•</span>
          VayuVision AI
        </div>

        <span>AI-powered weather intelligence</span>
      </footer>
    </div>
  );
}

// ============================================================
// START REACT
// ============================================================

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
