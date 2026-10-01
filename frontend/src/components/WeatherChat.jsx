import React, { useEffect, useRef, useState } from "react";

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
} from "lucide-react";

const SUGGESTIONS = [
  "Will it rain tomorrow?",
  "How windy will it be tomorrow?",
  "Should I go out on Monday?",
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
    <AIMessage
      from="user"
      timestamp={formatClock(message.timestamp)}
      copyText={message.content}
    >
      <span>{message.content}</span>
    </AIMessage>
  );
}

function AssistantMessage({ message }) {
  return (
    <AIMessage
      from="assistant"
      avatar={<SiriOrb size="28px" state="done" />}
      bubble={false}
      timestamp={formatClock(message.timestamp)}
      copyText={message.content}
    >
      <AIResponse text={message.content} />
    </AIMessage>
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

  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    const container = chatScrollRef.current;

    if (!container) {
      return;
    }

    requestAnimationFrame(() => {
      container.scrollTo({
        top: container.scrollHeight,
        behavior: "smooth",
      });
    });
  }, [messages.length, loadingChat]);

  function sendMessage(text) {
    const value = String(text || "").trim();

    if (!value || loadingChat) {
      return;
    }

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

    if (loadingChat) {
      return;
    }

    const value = question.trim();

    if (!value) {
      return;
    }

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
      {/* HEADER */}

      <header
        className="
          flex
          h-[64px]
          shrink-0
          items-center
          gap-3
          border-b
          border-border/60
          bg-background
          px-5
        "
      >
        <button
          type="button"
          onClick={() => setSidebarOpen((value) => !value)}
          className="
            rounded-lg
            p-2
            text-muted-foreground
            transition-colors
            hover:bg-muted
            hover:text-foreground
          "
          title="Chat options"
        >
          <PanelLeftOpen size={17} />
        </button>

        <div className="flex min-w-0 flex-1 items-center gap-3">
          <SiriOrb size="32px" state={loadingChat ? "thinking" : "idle"} />

          <div className="min-w-0">
            <h2 className="truncate text-sm font-semibold">{title}</h2>

            <p className="text-xs text-muted-foreground">
              AI weather intelligence
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
              transition-colors
              hover:bg-muted
              disabled:pointer-events-none
              disabled:opacity-50
            "
          >
            <Plus size={14} />
            New chat
          </button>
        )}
      </header>

      {/* CHAT MENU */}

      {sidebarOpen && (
        <div
          className="absolute inset-0 z-50 bg-black/10"
          onClick={() => setSidebarOpen(false)}
        >
          <div
            className="
              absolute
              left-4
              top-4
              w-64
              rounded-2xl
              border
              border-border/60
              bg-background
              p-4
              shadow-xl
            "
            onClick={(event) => event.stopPropagation()}
          >
            <div className="mb-4 flex items-center gap-2 text-sm font-medium">
              <Sparkles size={16} />
              WeatherGPT
            </div>

            <button
              type="button"
              onClick={() => {
                clearChat();
                setSidebarOpen(false);
              }}
              className="
                flex
                w-full
                items-center
                gap-2
                rounded-xl
                px-3
                py-2
                text-left
                text-sm
                transition
                hover:bg-muted
              "
            >
              <Plus size={15} />
              New chat
            </button>

            <div className="mt-4 rounded-xl bg-muted/60 p-3 text-xs text-muted-foreground">
              Your conversations are saved using the WeatherGPT session system.
            </div>
          </div>
        </div>
      )}

      {/* ======================================================
          THIS IS THE ONLY SCROLLING ELEMENT
      ====================================================== */}

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
            gap-5
            px-5
            py-6
          "
        >
          {/* EMPTY STATE */}

          {messages.length === 0 && !loadingHistory && (
            <div
              className="
                flex
                min-h-[430px]
                flex-1
                flex-col
                items-center
                justify-center
                gap-5
                py-10
                text-center
              "
            >
              <SiriOrb size="72px" state="idle" />

              <div>
                <h3 className="text-lg font-semibold">
                  Ask anything about the weather
                </h3>

                <p className="mx-auto mt-2 max-w-md text-sm text-muted-foreground">
                  Ask about rain, temperature, wind, travel conditions,
                  forecasts and more.
                </p>
              </div>

              <AISuggestions
                suggestions={SUGGESTIONS.map((label) => ({
                  label,
                }))}
                onSelect={(suggestion) => handleSuggestion(suggestion.label)}
              />
            </div>
          )}

          {/* HISTORY LOADING */}

          {loadingHistory && (
            <div className="flex min-h-[430px] flex-1 items-center justify-center">
              <div className="flex items-center gap-3 text-sm text-muted-foreground">
                <RefreshCw size={16} className="animate-spin" />
                Restoring conversation...
              </div>
            </div>
          )}

          {/* MESSAGES */}

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

          {/* THINKING */}

          {loadingChat && (
            <AIMessage
              from="assistant"
              avatar={<SiriOrb size="28px" state="thinking" />}
              bubble={false}
            >
              <AILoader
                label="WeatherGPT is thinking"
                showElapsed
                variant="dots"
              />
            </AIMessage>
          )}

          <div className="h-2 shrink-0" />
        </div>
      </div>

      {/* INPUT */}

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
          <form onSubmit={handleSubmit} className="relative">
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
                min-h-[60px]
                max-h-[140px]
                w-full
                resize-none
                rounded-2xl
                border
                border-border
                bg-background
                px-4
                py-4
                pr-14
                text-sm
                text-foreground
                outline-none
                transition
                placeholder:text-muted-foreground
                focus:border-ring
                focus:ring-2
                focus:ring-ring/20
                disabled:cursor-not-allowed
                disabled:opacity-60
              "
            />

            <button
              type="submit"
              disabled={loadingChat || !question.trim()}
              className="
                absolute
                bottom-3
                right-3
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
                disabled:opacity-40
              "
              title="Send message"
            >
              {loadingChat ? (
                <RefreshCw size={16} className="animate-spin" />
              ) : (
                <ArrowUp size={17} />
              )}
            </button>
          </form>

          <div className="pt-2 text-center text-xs text-muted-foreground">
            WeatherGPT • AI-powered weather intelligence
          </div>
        </div>
      </div>
    </section>
  );
}
