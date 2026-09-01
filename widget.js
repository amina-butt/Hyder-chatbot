/*!
 * Hyder Assistant — Standalone Chat Widget
 * Drop-in embeddable chat bubble with streaming responses.
 *
 * Usage:
 *   <link rel="stylesheet" href="widget.css">
 *   <script src="widget.js"></script>
 *
 * Optional config (set before this script loads):
 *   <script>
 *     window.HYDER_CHAT_CONFIG = {
 *       apiUrl: "http://localhost:8000/api/chat/stream",
 *       title: "Hyder Assistant"
 *     };
 *   </script>
 */
(function () {
  "use strict";

  var CONFIG = Object.assign(
    {
      apiUrl: "http://localhost:8000/api/chat/stream",
      title: "Hyder Assistant",
      storageKey: "session_id",
      greeting: "Hi! How can I help you today?",
    },
    window.HYDER_CHAT_CONFIG || {}
  );

  /* ----------------------------------------------------------------
   * 1. Session persistence
   * ---------------------------------------------------------------- */
  function getOrCreateSessionId() {
    try {
      var existing = localStorage.getItem(CONFIG.storageKey);
      if (existing) return existing;

      var fresh =
        window.crypto && typeof window.crypto.randomUUID === "function"
          ? window.crypto.randomUUID()
          : fallbackUUID();

      localStorage.setItem(CONFIG.storageKey, fresh);
      return fresh;
    } catch (err) {
      // localStorage unavailable (private mode, etc.) — use an in-memory id
      console.warn("[HyderChat] localStorage unavailable, using in-memory session id.", err);
      return fallbackUUID();
    }
  }

  function fallbackUUID() {
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
      var r = (Math.random() * 16) | 0;
      var v = c === "x" ? r : (r & 0x3) | 0x8;
      return v.toString(16);
    });
  }

  var sessionId = getOrCreateSessionId();

  /* ----------------------------------------------------------------
   * 2. Minimal Markdown -> HTML (bold + bullet lists only)
   * ---------------------------------------------------------------- */
  function escapeHtml(str) {
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function markdownToHtml(raw) {
    var text = escapeHtml(raw);

    // **bold**
    text = text.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

    // Bullet lines starting with "- "
    var lines = text.split("\n");
    var html = [];
    var inList = false;

    lines.forEach(function (line) {
      var bulletMatch = /^\s*-\s+(.*)$/.exec(line);
      if (bulletMatch) {
        if (!inList) {
          html.push("<ul>");
          inList = true;
        }
        html.push("<li>" + bulletMatch[1] + "</li>");
      } else {
        if (inList) {
          html.push("</ul>");
          inList = false;
        }
        if (line.trim() !== "") {
          html.push(line + "<br>");
        }
      }
    });

    if (inList) html.push("</ul>");

    return html.join("");
  }

  /* ----------------------------------------------------------------
   * 3. DOM construction
   * ---------------------------------------------------------------- */
  var root = document.createElement("div");
  root.className = "hyder-chat-widget";
  root.innerHTML =
    '<button type="button" class="hyder-chat-bubble-btn" aria-label="Open chat">' +
    '<svg viewBox="0 0 24 24"><path d="M4 4h16a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H8l-4 4V6a2 2 0 0 1 2-2z"/></svg>' +
    "</button>" +
    '<div class="hyder-chat-window" role="dialog" aria-label="' + CONFIG.title + '">' +
    '<div class="hyder-chat-header">' +
    '<p class="hyder-chat-title">' + escapeHtml(CONFIG.title) + "</p>" +
    '<button type="button" class="hyder-chat-close-btn" aria-label="Close chat">&times;</button>' +
    "</div>" +
    '<div class="hyder-chat-messages"></div>' +
    '<div class="hyder-chat-input-row">' +
    '<textarea class="hyder-chat-input" rows="1" placeholder="Type a message..."></textarea>' +
    '<button type="button" class="hyder-chat-send-btn" aria-label="Send message">' +
    '<svg viewBox="0 0 24 24"><path d="M2 21l21-9L2 3v7l15 2-15 2z"/></svg>' +
    "</button>" +
    "</div>" +
    "</div>";

  document.addEventListener("DOMContentLoaded", mount);
  if (document.readyState === "complete" || document.readyState === "interactive") {
    mount();
  }

  function mount() {
    if (document.body && !document.body.contains(root)) {
      document.body.appendChild(root);
      init();
    }
  }

  function init() {
    var bubbleBtn = root.querySelector(".hyder-chat-bubble-btn");
    var closeBtn = root.querySelector(".hyder-chat-close-btn");
    var windowEl = root.querySelector(".hyder-chat-window");
    var messagesEl = root.querySelector(".hyder-chat-messages");
    var inputEl = root.querySelector(".hyder-chat-input");
    var sendBtn = root.querySelector(".hyder-chat-send-btn");

    var isOpen = false;
    var isStreaming = false;

    bubbleBtn.addEventListener("click", toggleWindow);
    closeBtn.addEventListener("click", toggleWindow);

    inputEl.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    });

    inputEl.addEventListener("input", function () {
      inputEl.style.height = "auto";
      inputEl.style.height = Math.min(inputEl.scrollHeight, 90) + "px";
    });

    sendBtn.addEventListener("click", handleSend);

    function toggleWindow() {
      isOpen = !isOpen;
      windowEl.classList.toggle("hyder-chat-open", isOpen);
      if (isOpen) {
        if (!messagesEl.hasChildNodes()) {
          appendMessage("bot", CONFIG.greeting);
        }
        inputEl.focus();
      }
    }

    function appendMessage(role, text) {
      var bubble = document.createElement("div");
      bubble.className = "hyder-chat-msg hyder-chat-msg-" + role;
      bubble.innerHTML = markdownToHtml(text);
      messagesEl.appendChild(bubble);
      scrollToBottom();
      return bubble;
    }

    function appendTypingIndicator() {
      var bubble = document.createElement("div");
      bubble.className = "hyder-chat-msg-typing";
      bubble.innerHTML = "<span></span><span></span><span></span>";
      messagesEl.appendChild(bubble);
      scrollToBottom();
      return bubble;
    }

    function scrollToBottom() {
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function setStreamingState(active) {
      isStreaming = active;
      sendBtn.disabled = active;
      inputEl.disabled = active;
    }

    /* ----------------------------------------------------------------
     * 4. Send + stream handling
     * ---------------------------------------------------------------- */
    function handleSend() {
      var text = inputEl.value.trim();
      if (!text || isStreaming) return;

      appendMessage("user", text);
      inputEl.value = "";
      inputEl.style.height = "auto";

      streamReply(text);
    }

    /**
     * Parses one SSE line and returns the extracted data payload, or null
     * if the line should be skipped (blank line / ":" comment-ping / any
     * other non-"data:" field).
     */
    function parseSseLine(line) {
      // Skip blank lines (SSE event separators).
      if (line.trim() === "") return null;

      // Skip comment/keepalive lines (e.g., ": ping")
      if (line.charAt(0) === ":") return null;

      // Only process "data:" payload fields
      if (line.indexOf("data:") === 0) {
        var payload = line.slice(5).trim();

        // Ignore completion markers and empty objects
        if (payload === "[DONE]" || payload === "{}") {
          return null;
        }

        try {
          // Parse {"chunk": "..."} JSON payload
          var parsed = JSON.parse(payload);
          return parsed.chunk !== undefined ? parsed.chunk : (parsed.text || payload);
        } catch (e) {
          // Fallback if raw text is sent instead of JSON
          return payload;
        }
      }

      // Ignore event:, id:, retry: headers
      return null;
    }

    function streamReply(userText) {
      setStreamingState(true);
      var typingEl = appendTypingIndicator();
      var botBubble = null;
      var accumulated = "";
      var lineBuffer = "";

      fetch(CONFIG.apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          message: userText,
        }),
      })
        .then(function (response) {
          if (!response.ok || !response.body) {
            throw new Error("Chat request failed with status " + response.status);
          }

          typingEl.remove();
          botBubble = appendMessage("bot", "");

          var reader = response.body.getReader();
          var decoder = new TextDecoder("utf-8");

          function processBufferedLines(isFinal) {
            var lines = lineBuffer.split("\n");

            // The last element may be an incomplete line (no trailing "\n"
            // yet) — hold it back in the buffer unless this is the final
            // flush at stream end.
            lineBuffer = isFinal ? "" : lines.pop();

            var changed = false;
            lines.forEach(function (rawLine) {
              // Strip a trailing \r for CRLF-terminated streams.
              var line = rawLine.replace(/\r$/, "");
              var content = parseSseLine(line);
              if (content !== null) {
                accumulated += content;
                changed = true;
              }
            });

            if (changed) {
              botBubble.innerHTML = markdownToHtml(accumulated);
              scrollToBottom();
            }
          }

          function pump() {
            return reader.read().then(function (result) {
              if (result.done) {
                // Flush any trailing partial line still in the buffer.
                lineBuffer += decoder.decode();
                processBufferedLines(true);
                setStreamingState(false);
                return;
              }

              lineBuffer += decoder.decode(result.value, { stream: true });
              processBufferedLines(false);

              return pump();
            });
          }

          return pump();
        })
        .catch(function (err) {
          console.error("[HyderChat] stream error:", err);
          if (typingEl.isConnected) typingEl.remove();
          if (!botBubble) {
            appendMessage("bot", "Sorry, something went wrong. Please try again.");
          } else if (!accumulated) {
            botBubble.innerHTML = markdownToHtml(
              "Sorry, something went wrong. Please try again."
            );
          }
          setStreamingState(false);
        });
    }
  }
})();