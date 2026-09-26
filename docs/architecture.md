# WeatherGPT MVP flow

User -> React -> Express -> LangGraph

LangGraph nodes:
1. Query understanding
2. Weather data retrieval
3. Data processing
4. Risk analysis via Random Forest service
5. Decision/alert logic
6. Response generation

External systems:
- Open-Meteo for current/forecast data in MVP
- MongoDB as the persistence layer to add next
- Gemini as optional LLM layer to add next
- Firebase Cloud Messaging for production notifications to add next
