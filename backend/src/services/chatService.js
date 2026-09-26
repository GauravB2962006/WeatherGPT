import { getDatabase } from "../db.js";

// MongoDB collection used for chat history
const COLLECTION_NAME = "conversations";

// ============================================================
// Save conversation
// ============================================================

export async function saveConversation({
  sessionId,
  city,
  userMessage,
  assistantMessage,
  targetDate = null,
  forecastIndex = null,
  intent = null,
  riskLevel = null,
}) {
  const db = await getDatabase();

  const collection = db.collection(COLLECTION_NAME);

  const now = new Date();

  // Store the user's message
  await collection.insertOne({
    sessionId,
    role: "user",
    content: userMessage,
    city,
    createdAt: now,
  });

  // Store WeatherGPT's response
  await collection.insertOne({
    sessionId,
    role: "assistant",
    content: assistantMessage,
    city,

    // Weather-related metadata
    targetDate,
    forecastIndex,
    intent,
    riskLevel,

    createdAt: new Date(),
  });

  console.log(`Conversation saved: ${sessionId}`);
}

// ============================================================
// Get conversation history
// ============================================================

export async function getConversation(sessionId) {
  const db = await getDatabase();

  const collection = db.collection(COLLECTION_NAME);

  const messages = await collection
    .find({ sessionId })
    .sort({ createdAt: 1 })
    .toArray();

  return messages.map((message) => ({
    role: message.role,
    content: message.content,

    // Keep metadata available for future use
    ...(message.role === "assistant"
      ? {
          targetDate: message.targetDate ?? null,

          forecastIndex: message.forecastIndex ?? null,

          intent: message.intent ?? null,

          riskLevel: message.riskLevel ?? null,
        }
      : {}),
  }));
}
