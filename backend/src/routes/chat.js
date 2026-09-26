import { Router } from "express";
import { saveConversation, getConversation } from "../services/chatService.js";

const router = Router();

// ============================================================
// POST /api/chat
// ============================================================

router.post("/", async (req, res) => {
  try {
    const { message, city, history = [], sessionId } = req.body || {};

    // ----------------------------------------------------------
    // Validate message
    // ----------------------------------------------------------

    if (!message || !message.trim()) {
      return res.status(400).json({
        error: "message is required",
      });
    }

    // ----------------------------------------------------------
    // Session ID
    // ----------------------------------------------------------

    const currentSessionId = sessionId || `session-${Date.now()}`;

    // ----------------------------------------------------------
    // AI service
    // ----------------------------------------------------------

    const aiUrl = process.env.AI_SERVICE_URL || "http://localhost:8002";

    // ----------------------------------------------------------
    // Send conversation history to LangGraph
    // ----------------------------------------------------------

    const response = await fetch(`${aiUrl}/chat`, {
      method: "POST",

      headers: {
        "content-type": "application/json",
      },

      body: JSON.stringify({
        message: message.trim(),

        city: city || null,

        history: Array.isArray(history) ? history : [],
      }),
    });

    const data = await response.json();

    // ----------------------------------------------------------
    // AI service error
    // ----------------------------------------------------------

    if (!response.ok) {
      console.error("AI service error:", data);

      return res.status(response.status).json(data);
    }

    // ----------------------------------------------------------
    // Save conversation to MongoDB
    // ----------------------------------------------------------

    try {
      await saveConversation({
        sessionId: currentSessionId,

        city: city || "Nashik",

        userMessage: message.trim(),

        assistantMessage: data.response || "",

        targetDate: data.target_date || null,

        forecastIndex: data.forecast_index ?? null,

        intent: data.intent || null,

        riskLevel: data.risk?.risk_level || null,
      });
    } catch (mongoError) {
      // Don't break the AI response if MongoDB
      // happens to fail.

      console.error("MongoDB save error:", mongoError);
    }

    // ----------------------------------------------------------
    // Return response
    // ----------------------------------------------------------

    return res.json({
      ...data,

      sessionId: currentSessionId,
    });
  } catch (error) {
    console.error("Chat route error:", error);

    return res.status(502).json({
      error: error.message || "AI service unavailable",
    });
  }
});

// ============================================================
// GET /api/chat/history/:sessionId
// ============================================================

router.get("/history/:sessionId", async (req, res) => {
  try {
    const sessionId = req.params.sessionId;

    if (!sessionId) {
      return res.status(400).json({
        error: "sessionId is required",
      });
    }

    const conversation = await getConversation(sessionId);

    return res.json(conversation);
  } catch (error) {
    console.error("History error:", error);

    return res.status(500).json({
      error: error.message || "Could not load conversation",
    });
  }
});

export default router;
