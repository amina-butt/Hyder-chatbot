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
      audioApiUrl: "http://localhost:8000/api/chat/audio",
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
    '<button type="button" class="hyder-chat-mic-btn" aria-label="Record voice message" aria-pressed="false">' +
    '<svg viewBox="0 0 24 24"><path d="M12 15a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3z"/>' +
    '<path d="M19 11a1 1 0 0 0-2 0 5 5 0 0 1-10 0 1 1 0 0 0-2 0 7 7 0 0 0 6 6.92V20H9a1 1 0 0 0 0 2h6a1 1 0 0 0 0-2h-2v-2.08A7 7 0 0 0 19 11z"/>' +
    "</svg>" +
    "</button>" +
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
    var micBtn = root.querySelector(".hyder-chat-mic-btn");
    var inputRowEl = root.querySelector(".hyder-chat-input-row");

    var isOpen = false;
    var isStreaming = false;

    /* ---------------------------------------------------------------
     * Voice recording state
     * --------------------------------------------------------------- */
    var isRecording = false;
    var mediaRecorder = null;
    var mediaStream = null;
    var audioChunks = [];
    var recordingStartedAt = 0;
    var recordingTimerId = null;
    var recordingIndicatorEl = null;

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
    micBtn.addEventListener("click", handleMicClick);

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
      micBtn.disabled = active;
    }

    /* ----------------------------------------------------------------
     * 4. Send + stream handling
     * ---------------------------------------------------------------- */
    function handleSend() {
      var text = inputEl.value.trim();
      if (!text || isStreaming || isRecording) return;

      appendMessage("user", text);
      inputEl.value = "";
      inputEl.style.height = "auto";

      streamReply(text);
    }

    /**
     * Reads an SSE (EventSource-style) response body, dispatching
     * onEvent(eventName, rawDataString) for each complete event. Tracks
     * the "event:" field (defaulting to "message" per the SSE spec when
     * absent) instead of treating every "data:" line the same way,
     * because the audio endpoint sends a distinct "transcript" event
     * ahead of the normal "message"/"error"/"done" events used by both
     * endpoints.
     */
    function consumeSseResponse(response, onEvent) {
      var reader = response.body.getReader();
      var decoder = new TextDecoder("utf-8");
      var lineBuffer = "";
      var currentEvent = "message";
      var dataLines = [];

      function dispatch() {
        if (dataLines.length) {
          onEvent(currentEvent, dataLines.join("\n"));
        }
        currentEvent = "message";
        dataLines = [];
      }

      function feedLine(rawLine) {
        var line = rawLine.replace(/\r$/, "");
        if (line === "") {
          dispatch();
          return;
        }
        if (line.charAt(0) === ":") return; // comment/keepalive
        var idx = line.indexOf(":");
        if (idx === -1) return;
        var field = line.slice(0, idx);
        var value = line.slice(idx + 1).replace(/^ /, "");
        if (field === "event") currentEvent = value;
        else if (field === "data") dataLines.push(value);
        // id:, retry: intentionally ignored
      }

      function processBuffer(isFinal) {
        var lines = lineBuffer.split("\n");
        lineBuffer = isFinal ? "" : lines.pop();
        lines.forEach(feedLine);
      }

      function pump() {
        return reader.read().then(function (result) {
          if (result.done) {
            lineBuffer += decoder.decode();
            processBuffer(true);
            dispatch(); // flush a trailing event with no closing blank line
            return;
          }
          lineBuffer += decoder.decode(result.value, { stream: true });
          processBuffer(false);
          return pump();
        });
      }

      return pump();
    }

    function safeJsonParse(raw) {
      try {
        return JSON.parse(raw);
      } catch (e) {
        return null;
      }
    }

    /**
     * Drives a chat SSE response (from either /api/chat/stream or
     * /api/chat/audio) into the message list: creates the bot bubble on
     * first content, streams "message" chunks into it, surfaces "error"
     * events as bot text, and — when onTranscript is given — fires it
     * once for the audio endpoint's "transcript" event so the caller can
     * render the user's own bubble with what was actually said.
     */
    function renderChatStream(response, typingEl, onTranscript) {
      if (!response.ok || !response.body) {
        throw new Error("Chat request failed with status " + response.status);
      }

      var botBubble = null;
      var accumulated = "";

      function ensureBotBubble() {
        if (typingEl.isConnected) typingEl.remove();
        if (!botBubble) botBubble = appendMessage("bot", "");
        return botBubble;
      }

      return consumeSseResponse(response, function (eventName, dataStr) {
        if (eventName === "transcript") {
          var parsed = safeJsonParse(dataStr);
          if (parsed && parsed.text && onTranscript) onTranscript(parsed.text);
          return;
        }

        if (eventName === "error") {
          var errParsed = safeJsonParse(dataStr);
          var message =
            (errParsed && errParsed.message) ||
            "Sorry, something went wrong. Please try again.";
          ensureBotBubble().innerHTML = markdownToHtml(accumulated + message);
          scrollToBottom();
          return;
        }

        if (dataStr === "[DONE]" || dataStr === "{}") return;

        var chunkParsed = safeJsonParse(dataStr);
        var content = chunkParsed
          ? chunkParsed.chunk !== undefined
            ? chunkParsed.chunk
            : chunkParsed.text
          : dataStr;
        if (!content) return;

        accumulated += content;
        ensureBotBubble().innerHTML = markdownToHtml(accumulated);
        scrollToBottom();
      });
    }

    function streamReply(userText) {
      setStreamingState(true);
      var typingEl = appendTypingIndicator();

      fetch(CONFIG.apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          message: userText,
        }),
      })
        .then(function (response) {
          return renderChatStream(response, typingEl, null);
        })
        .catch(function (err) {
          console.error("[HyderChat] stream error:", err);
          if (typingEl.isConnected) typingEl.remove();
          appendMessage("bot", "Sorry, something went wrong. Please try again.");
        })
        .finally(function () {
          setStreamingState(false);
        });
    }

    /* ----------------------------------------------------------------
     * 5. Voice recording + send
     * ---------------------------------------------------------------- */
    function pickAudioMimeType() {
      var candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/wav"];
      for (var i = 0; i < candidates.length; i++) {
        if (window.MediaRecorder && MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(candidates[i])) {
          return candidates[i];
        }
      }
      return ""; // let the browser pick its own default
    }

    function formatElapsed(ms) {
      var totalSeconds = Math.floor(ms / 1000);
      var m = Math.floor(totalSeconds / 60);
      var s = totalSeconds % 60;
      return m + ":" + (s < 10 ? "0" : "") + s;
    }

    function showRecordingIndicator() {
      recordingIndicatorEl = document.createElement("span");
      recordingIndicatorEl.className = "hyder-chat-recording-indicator";
      recordingIndicatorEl.innerHTML = '<span class="hyder-chat-recording-dot"></span><span>0:00</span>';
      inputRowEl.insertBefore(recordingIndicatorEl, inputEl);
      inputEl.style.display = "none";

      recordingTimerId = setInterval(function () {
        if (!recordingIndicatorEl) return;
        var label = recordingIndicatorEl.querySelector("span:last-child");
        if (label) label.textContent = formatElapsed(Date.now() - recordingStartedAt);
      }, 500);
    }

    function hideRecordingIndicator() {
      if (recordingTimerId) {
        clearInterval(recordingTimerId);
        recordingTimerId = null;
      }
      if (recordingIndicatorEl) {
        recordingIndicatorEl.remove();
        recordingIndicatorEl = null;
      }
      inputEl.style.display = "";
    }

    function setRecordingUiState(active) {
      isRecording = active;
      micBtn.classList.toggle("hyder-chat-mic-active", active);
      micBtn.setAttribute("aria-pressed", active ? "true" : "false");
      sendBtn.disabled = active || isStreaming;
      if (active) {
        showRecordingIndicator();
      } else {
        hideRecordingIndicator();
      }
    }

    function stopMediaTracks() {
      if (mediaStream) {
        mediaStream.getTracks().forEach(function (track) {
          track.stop();
        });
        mediaStream = null;
      }
    }

    function handleMicClick() {
      if (isStreaming) return;
      if (isRecording) {
        stopRecording();
      } else {
        startRecording();
      }
    }

    function startRecording() {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !window.MediaRecorder) {
        window.alert("Voice input isn't supported in this browser.");
        return;
      }

      navigator.mediaDevices
        .getUserMedia({ audio: true })
        .then(function (stream) {
          mediaStream = stream;
          audioChunks = [];

          var mimeType = pickAudioMimeType();
          try {
            mediaRecorder = mimeType ? new MediaRecorder(stream, { mimeType: mimeType }) : new MediaRecorder(stream);
          } catch (err) {
            console.error("[HyderChat] could not start MediaRecorder:", err);
            window.alert("Couldn't start audio recording on this device.");
            stopMediaTracks();
            return;
          }

          mediaRecorder.addEventListener("dataavailable", function (e) {
            if (e.data && e.data.size > 0) audioChunks.push(e.data);
          });

          mediaRecorder.addEventListener("stop", function () {
            stopMediaTracks();
            var blobType = mediaRecorder.mimeType || mimeType || "audio/webm";
            var blob = new Blob(audioChunks, { type: blobType });
            audioChunks = [];
            if (blob.size > 0) {
              sendAudioMessage(blob, blobType);
            }
          });

          mediaRecorder.start();
          recordingStartedAt = Date.now();
          setRecordingUiState(true);
        })
        .catch(function (err) {
          console.error("[HyderChat] microphone access denied or unavailable:", err);
          window.alert("Microphone access is required for voice messages. Please allow microphone access and try again.");
        });
    }

    function stopRecording() {
      if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
      }
      setRecordingUiState(false);
    }

    function sendAudioMessage(blob, mimeType) {
      setStreamingState(true);
      var typingEl = appendTypingIndicator();

      var extension = mimeType.indexOf("wav") !== -1 ? "wav" : mimeType.indexOf("ogg") !== -1 ? "ogg" : "webm";
      var formData = new FormData();
      formData.append("session_id", sessionId);
      formData.append("file", blob, "voice-message." + extension);

      fetch(CONFIG.audioApiUrl, {
        method: "POST",
        body: formData,
      })
        .then(function (response) {
          return renderChatStream(response, typingEl, function (transcript) {
            appendMessage("user", transcript);
          });
        })
        .catch(function (err) {
          console.error("[HyderChat] audio stream error:", err);
          if (typingEl.isConnected) typingEl.remove();
          appendMessage("bot", "Sorry, something went wrong sending your voice message. Please try again.");
        })
        .finally(function () {
          setStreamingState(false);
        });
    }
  }
})();