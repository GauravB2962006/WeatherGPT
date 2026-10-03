import React, { useEffect, useRef } from "react";

import AIMessage from "@/components/smoothui/ai-message";
import AIResponse from "@/components/smoothui/ai-response";
import AILoader from "@/components/smoothui/ai-loader";
import AISuggestions from "@/components/smoothui/ai-suggestions";
import SiriOrb from "@/components/smoothui/siri-orb";

import {
  ArrowUp,
  Plus,
  PanelLeftOpen,
  RefreshCw,
  Sparkles,
  CloudSun,
  Copy,
} from "lucide-react";

const SUGGESTIONS = [
  {
    label: "Will it rain tomorrow?",
    icon: "🌧️",
  },
  {
    label: "How's the weather on Monday?",
    icon: "📅",
  },
  {
    label: "Is tomorrow good for a trip?",
    icon: "✈️",
  },
  {
    label: "How hot will it be tomorrow?",
    icon: "🌡️",
  },
];

function formatClock(timestamp) {
  if (!timestamp) return "";

  const date = new Date(timestamp);

  if (Number.isNaN(date.getTime())) {
    return "";
  }

  return `${String(date.getHours()).padStart(2, "0")}:${String(
    date.getMinutes(),
  ).padStart(2, "0")}`;
}

function UserMessage({ message }) {
  return (
    <div className="flex w-full justify-end">
      <div className="flex max-w-[82%] flex-col items-end gap-1.5">
        <AIMessage
          from="user"
          timestamp={formatClock(message.timestamp)}
          copyText={message.content}
        >
          <span className="leading-6">{message.content}</span>
        </AIMessage>
      </div>
    </div>
  );
}

function AssistantMessage({ message }) {
  return (
    <div className="group flex w-full items-start gap-3">
      <div className="mt-1 shrink-0">
        <SiriOrb size="30px" state="done" />
      </div>

      <div className="min-w-0 flex-1">
        <div className="mb-1 flex items-center gap-2">
          <span className="text-xs font-semibold text-foreground">
            WeatherGPT
          </span>

          <span className="text-[11px] text-muted-foreground">
            {formatClock(message.timestamp)}
          </span>
        </div>

        <AIMessage from="assistant" bubble={false} copyText={message.content}>
          <AIResponse text={message.content} />
        </AIMessage>
      </div>
    </div>
  );
}

export default function WeatherChat({
  messages = [],
  question,
  setQuestion,
  loadingChat,
  loadingHistory,
  askWeatherGPT,
  clearChat,
  title = "WeatherGPT",
}) {
  const chatScrollRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    const container = chatScrollRef.current;

    if (!container) return;

    requestAnimationFrame(() => {
      container.scrollTo({
        top: container.scrollHeight,
        behavior: "smooth",
      });
    });
  }, [messages.length, loadingChat]);

  function sendMessage(text) {
    const value = String(text || "").trim();

    if (!value || loadingChat) return;

    setQuestion(value);

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        askWeatherGPT({
          preventDefault() {},
        });
      });
    });
  }

  function handleSubmit(event) {
    event.preventDefault();

    if (loadingChat) return;

    const value = question.trim();

    if (!value) return;

    sendMessage(value);
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();

      if (!loadingChat && question.trim()) {
        sendMessage(question);
      }
    }
  }

  function handleSuggestion(text) {
    sendMessage(text);
  }

  return (
    <section
      className="
        relative
        flex
        h-[620px]
        max-h-[620px]
        min-h-0
        min-w-0
        flex-1
        flex-col
        overflow-hidden
        rounded-3xl
        border
        border-border/60
        bg-background
        shadow-sm
      "
    >
      {/* ============================================================
          HEADER
      ============================================================ */}

      <header
        className="
          flex
          h-[68px]
          shrink-0
          items-center
          gap-3
          border-b
          border-border/60
          bg-background/95
          px-5
          backdrop-blur
        "
      >
        <button
          type="button"
          className="
            flex
            h-9
            w-9
            items-center
            justify-center
            rounded-xl
            border
            border-border/60
            text-muted-foreground
            transition
            hover:bg-muted
            hover:text-foreground
          "
          title="Chat options"
        >
          <PanelLeftOpen size={17} />
        </button>

        <div
          className="
            flex
            min-w-0
            flex-1
            items-center
            gap-3
          "
        >
          <div className="relative">
            <SiriOrb size="34px" state={loadingChat ? "thinking" : "idle"} />

            <span
              className="
                absolute
                bottom-0
                right-0
                h-2.5
                w-2.5
                rounded-full
                border-2
                border-background
                bg-emerald-500
              "
            />
          </div>

          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h2 className="truncate text-sm font-semibold">{title}</h2>

              <span
                className="
                  rounded-full
                  bg-muted
                  px-2
                  py-0.5
                  text-[10px]
                  font-medium
                  text-muted-foreground
                "
              >
                AI
              </span>
            </div>

            <p className="truncate text-xs text-muted-foreground">
              Ask anything about the weather
            </p>
          </div>
        </div>

        {messages.length > 0 && (
          <button
            type="button"
            onClick={clearChat}
            disabled={loadingChat}
            className="
              flex
              items-center
              gap-2
              rounded-xl
              border
              border-border/60
              bg-background
              px-3
              py-2
              text-xs
              font-medium
              transition
              hover:bg-muted
              disabled:pointer-events-none
              disabled:opacity-50
            "
          >
            <Plus size={14} />
            <span className="hidden sm:inline">New chat</span>
          </button>
        )}
      </header>

      {/* ============================================================
          CHAT AREA
      ============================================================ */}

      <div
        ref={chatScrollRef}
        className="
          min-h-0
          flex-1
          overflow-x-hidden
          overflow-y-auto
          overscroll-contain
          scroll-smooth
        "
        style={{
          scrollbarWidth: "thin",
        }}
      >
        <div
          className="
            mx-auto
            flex
            min-h-full
            w-full
            max-w-3xl
            flex-col
            gap-6
            px-5
            py-7
          "
        >
          {/* ========================================================
              EMPTY STATE
          ======================================================== */}

          {messages.length === 0 && !loadingHistory && (
            <div
              className="
                flex
                min-h-[430px]
                flex-1
                flex-col
                items-center
                justify-center
                px-3
                py-10
                text-center
              "
            >
              <div
                className="
                  mb-5
                  flex
                  h-16
                  w-16
                  items-center
                  justify-center
                  rounded-2xl
                  border
                  border-border/60
                  bg-muted/40
                  shadow-sm
                "
              >
                <SiriOrb size="54px" state="idle" />
              </div>

              <div>
                <div
                  className="
                    mb-2
                    flex
                    items-center
                    justify-center
                    gap-2
                    text-xs
                    font-medium
                    text-muted-foreground
                  "
                >
                  <Sparkles size={14} />
                  WeatherGPT AI
                </div>

                <h3 className="text-xl font-semibold tracking-tight">
                  How can I help with the weather?
                </h3>

                <p
                  className="
                    mx-auto
                    mt-2
                    max-w-md
                    text-sm
                    leading-6
                    text-muted-foreground
                  "
                >
                  Ask about forecasts, rain, temperature, wind, travel
                  conditions, or any other weather-related question.
                </p>
              </div>

              <div className="mt-7 w-full max-w-2xl">
                <AISuggestions
                  suggestions={SUGGESTIONS.map((item) => ({
                    label: `${item.icon}  ${item.label}`,
                  }))}
                  onSelect={(suggestion) => {
                    handleSuggestion(
                      suggestion.label.replace(/^(🌧️|📅|✈️|🌡️)\s+/, ""),
                    );
                  }}
                />
              </div>
            </div>
          )}

          {/* ========================================================
              HISTORY LOADING
          ======================================================== */}

          {loadingHistory && (
            <div className="flex min-h-[430px] flex-1 items-center justify-center">
              <div
                className="
                  flex
                  items-center
                  gap-3
                  rounded-xl
                  border
                  border-border/60
                  bg-muted/30
                  px-4
                  py-3
                  text-sm
                  text-muted-foreground
                "
              >
                <RefreshCw size={16} className="animate-spin" />
                Restoring conversation...
              </div>
            </div>
          )}

          {/* ========================================================
              MESSAGES
          ======================================================== */}

          {messages.map((message, index) => (
            <div
              key={`${message.timestamp || "message"}-${index}`}
              className="w-full"
            >
              {message.role === "user" ? (
                <UserMessage message={message} />
              ) : (
                <AssistantMessage message={message} />
              )}
            </div>
          ))}

          {/* ========================================================
              THINKING
          ======================================================== */}

          {loadingChat && (
            <div className="flex items-start gap-3">
              <div className="mt-1 shrink-0">
                <SiriOrb size="30px" state="thinking" />
              </div>

              <AIMessage from="assistant" bubble={false}>
                <AILoader
                  label="WeatherGPT is thinking"
                  showElapsed
                  variant="dots"
                />
              </AIMessage>
            </div>
          )}

          <div className="h-1 shrink-0" />
        </div>
      </div>

      {/* ============================================================
          INPUT AREA
      ============================================================ */}

      <div
        className="
          shrink-0
          border-t
          border-border/60
          bg-background
          px-4
          py-4
        "
      >
        <div className="mx-auto w-full max-w-3xl">
          <form onSubmit={handleSubmit}>
            <div
              className="
                relative
                overflow-hidden
                rounded-2xl
                border
                border-border/70
                bg-background
                shadow-sm
                transition
                focus-within:border-ring
                focus-within:ring-2
                focus-within:ring-ring/10
              "
            >
              <textarea
                ref={inputRef}
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                onKeyDown={handleKeyDown}
                disabled={loadingChat}
                maxLength={2000}
                rows={1}
                placeholder="Ask anything about the weather..."
                className="
                  min-h-[58px]
                  max-h-[140px]
                  w-full
                  resize-none
                  bg-transparent
                  px-4
                  py-4
                  pr-14
                  text-sm
                  leading-6
                  text-foreground
                  outline-none
                  placeholder:text-muted-foreground
                  disabled:cursor-not-allowed
                  disabled:opacity-60
                "
              />

              <button
                type="submit"
                disabled={loadingChat || !question.trim()}
                className="
                  absolute
                  bottom-2.5
                  right-2.5
                  flex
                  h-9
                  w-9
                  items-center
                  justify-center
                  rounded-xl
                  bg-foreground
                  text-background
                  transition
                  hover:opacity-80
                  disabled:cursor-not-allowed
                  disabled:opacity-30
                "
                title="Send message"
              >
                {loadingChat ? (
                  <RefreshCw size={16} className="animate-spin" />
                ) : (
                  <ArrowUp size={17} />
                )}
              </button>
            </div>
          </form>

          <div
            className="
              flex
              items-center
              justify-center
              gap-2
              pt-2.5
              text-[11px]
              text-muted-foreground
            "
          >
            <CloudSun size={13} />
            <span>WeatherGPT · AI-powered weather intelligence</span>
          </div>
        </div>
      </div>
    </section>
  );
}
