
import os
from flask import Flask, request, jsonify, render_template_string
from groq import Groq

app = Flask(__name__)

# SAM-JARVIS Configuration
APP_NAME = "SAM-JARVIS"
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=API_KEY) if API_KEY else None

SYSTEM_PROMPT = """
You are SAM-JARVIS, a friendly personal AI assistant.
Your user's name is Sambit.
Address him as Sambit or bro. Never call him boss.

Be helpful, intelligent, friendly, and concise.

You are running on a cloud server.
You cannot directly access the user's microphone,
Windows computer, or ESP32 hardware.

Never claim to have turned on an LED, buzzer,
or other hardware unless a local agent confirms it.
"""

# Web dashboard
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SAM-JARVIS</title>

<style>
* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background: #07111f;
    color: #eaf5ff;
    font-family: Arial, sans-serif;
}

header {
    padding: 22px;
    background: #101f33;
    border-bottom: 1px solid #25415e;
}

h1 {
    margin: 0;
    color: #35d5ff;
}

.subtitle {
    color: #9bb0c7;
    margin-top: 8px;
}

.container {
    max-width: 850px;
    margin: 30px auto;
    padding: 15px;
}

.panel {
    background: #101f33;
    border: 1px solid #25415e;
    border-radius: 16px;
    padding: 20px;
}

#chat {
    height: 55vh;
    min-height: 300px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 15px 0;
}

.message {
    padding: 13px;
    border-radius: 12px;
    max-width: 90%;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    line-height: 1.5;
}

.user {
    background: #12486a;
    align-self: flex-end;
}

.bot {
    background: #20344b;
    align-self: flex-start;
}

form {
    display: flex;
    gap: 10px;
}

input {
    flex: 1;
    min-width: 0;
    padding: 14px;
    border-radius: 10px;
    border: 1px solid #25415e;
    background: #07111f;
    color: white;
    font-size: 16px;
}

button {
    padding: 14px 20px;
    background: #35d5ff;
    border: none;
    border-radius: 10px;
    font-weight: bold;
    cursor: pointer;
}

button:disabled {
    opacity: 0.5;
}

#status {
    color: #9bb0c7;
    margin: 10px 0;
}

.note {
    color: #9bb0c7;
    font-size: 13px;
    line-height: 1.5;
    margin-top: 18px;
}
</style>
</head>

<body>

<header>
    <h1>SAM-JARVIS</h1>
    <div class="subtitle">
        Personal AI Assistant | Powered by Groq
    </div>
</header>

<div class="container">
    <div class="panel">
        <h2>Chat with Jarvis</h2>

        <div id="status">System online</div>

        <div id="chat">
            <div class="message bot">
                Hey Sambit! Jarvis is online. How can I help you?
            </div>
        </div>

        <form id="chatForm">
            <input
                id="message"
                placeholder="Ask Jarvis anything..."
                autocomplete="off"
                required
            >
            <button id="send" type="submit">Send</button>
        </form>

        <div class="note">
            Cloud chat is active. Microphone, wake-word detection,
            and direct ESP32 control require a Windows local agent
            running on your network.
        </div>
    </div>
</div>

<script>
const chat = document.getElementById("chat");
const form = document.getElementById("chatForm");
const input = document.getElementById("message");
const send = document.getElementById("send");
const status = document.getElementById("status");

function addMessage(type, text) {
    const div = document.createElement("div");
    div.className = "message " + type;
    div.textContent = text;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
}

form.addEventListener("submit", async function(event) {
    event.preventDefault();

    const message = input.value.trim();
    if (!message) return;

    addMessage("user", message);
    input.value = "";
    send.disabled = true;
    status.textContent = "Jarvis is thinking...";

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: message
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Request failed");
        }

        addMessage("bot", data.reply);

    } catch (error) {
        addMessage("bot", "Error: " + error.message);
    } finally {
        send.disabled = false;
        status.textContent = "System online";
        input.focus();
    }
});
</script>

</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/health")
def health():
    return jsonify({
        "status": "online",
        "app": APP_NAME,
        "model": MODEL
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    if client is None:
        return jsonify({
            "error": "GROQ_API_KEY is missing. Add it in Render Environment Variables."
        }), 500

    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({
            "error": "Please enter a message."
        }), 400

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": message
                }
            ],
            temperature=0.7,
            max_tokens=1024
        )

        reply = response.choices[0].message.content

        return jsonify({
            "reply": reply or "I couldn't generate a response.",
            "app": APP_NAME
        })

    except Exception as e:
        app.logger.exception("Groq API error")
        return jsonify({
            "error": "Groq API error: " + str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
