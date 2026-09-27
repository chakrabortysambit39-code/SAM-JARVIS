"""
SAM-JARVIS 2.0
Hands-free "Hey Jarvis" wake word + Groq chat + ESP32 controls.

Install:
  py -m pip install flask requests groq openwakeword sounddevice numpy SpeechRecognition pyttsx3

Set Groq key in PowerShell (keep it private):
  $env:GROQ_API_KEY="YOUR_GROQ_API_KEY"

Run:
  py app.py
Open:
  http://127.0.0.1:5000

Notes:
- Wake-word detection runs locally on your PC.
- After wake-word activation, Google Speech Recognition transcribes the command online.
- Windows may ask for microphone permission.
- Keep your Groq API key private; never paste it into the HTML/dashboard.
"""

import os
import re
import time
import threading
import queue
import webbrowser

import requests
import numpy as np
import sounddevice as sd
import speech_recognition as sr
import pyttsx3
from flask import Flask, request, jsonify, render_template_string

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from openwakeword.model import Model
except ImportError:
    Model = None


# ---------------- CONFIG ----------------
APP_NAME = "SAM-JARVIS"
USER_NAME = "Sambit"
ESP32_BASE_URL = "http://192.168.1.10"
GROQ_MODEL = "openai/gpt-oss-120b"
HOST = "127.0.0.1"
PORT = 5000

SAMPLE_RATE = 16000
FRAME_SIZE = 1280                 # 80 ms at 16 kHz
WAKE_THRESHOLD = 0.55
WAKE_COOLDOWN_SECONDS = 2.0
COMMAND_TIMEOUT = 7
PHRASE_LIMIT = 9

SYSTEM_PROMPT = f"""
You are SAM-JARVIS, a friendly personal AI assistant for {USER_NAME}.
Address the user as {USER_NAME} or "bro" naturally.
NEVER call the user "Boss", "sir", "master", or "chief".
Be warm, helpful, concise, and conversational.
You can explain that physical devices are controlled only when the local app confirms it.
Do not claim to have controlled hardware unless the app actually reports success.
"""

app = Flask(__name__)
chat_history = []
history_lock = threading.Lock()

status_lock = threading.Lock()
assistant_status = {
    "mode": "Starting",
    "message": "Starting SAM-JARVIS…",
    "last_heard": "",
    "last_reply": "",
    "wake_enabled": True,
}

wake_stop_event = threading.Event()
wake_thread = None
audio_stream = None
tts_lock = threading.Lock()


# ---------------- SPEECH ----------------
def speak(text):
    """Speak text on the PC. TTS is initialized in this function for thread safety."""
    text = str(text or "").strip()
    if not text:
        return
    # Never let the assistant address the user as Boss, including generated replies.
    text = re.sub(r"\bboss\b", USER_NAME, text, flags=re.IGNORECASE)
    with tts_lock:
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as exc:
            print("TTS error:", exc)


def set_status(mode=None, message=None, last_heard=None, last_reply=None):
    with status_lock:
        if mode is not None:
            assistant_status["mode"] = mode
        if message is not None:
            assistant_status["message"] = message
        if last_heard is not None:
            assistant_status["last_heard"] = last_heard
        if last_reply is not None:
            assistant_status["last_reply"] = last_reply


def get_status():
    with status_lock:
        return dict(assistant_status)


# ---------------- ESP32 ----------------
def esp32_request(path, params=None):
    try:
        response = requests.get(
            f"{ESP32_BASE_URL}{path}",
            params=params or {},
            timeout=3
        )
        response.raise_for_status()
        return True, response.text.strip() or "ESP32 accepted the command."
    except requests.RequestException as exc:
        return False, f"I couldn't reach the ESP32 at {ESP32_BASE_URL}. Check Wi-Fi and the ESP32 IP."


def hardware_command(text):
    """Return a reply for recognized hardware commands, otherwise None."""
    t = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    t = re.sub(r"\s+", " ", t).strip()

    # Normalize common speech-to-text mishearings.
    t = re.sub(r"\b(buser|buzer|buzzle|busser|buzzard)\b", "buzzer", t)
    t = re.sub(r"\b(lights|light|lamp|bulb)\b", "led", t)

    if any(x in t for x in ("all off", "turn everything off", "switch everything off", "turn all off")):
        ok, detail = esp32_request("/alloff")
        return "All connected devices are switched off." if ok else detail

    # LED
    if "led" in t:
        if any(x in t for x in ("turn on", "switch on", "switch the", "activate", "enable", "start")) and not any(x in t for x in ("turn off", "switch off", "deactivate", "disable", "stop")):
            ok, detail = esp32_request("/led1", {"state": "on"})
            return "LED is on." if ok else detail
        if any(x in t for x in ("turn off", "switch off", "deactivate", "disable", "stop")):
            ok, detail = esp32_request("/led1", {"state": "off"})
            return "LED is off." if ok else detail

    # Buzzer
    if "buzzer" in t:
        if any(x in t for x in ("turn on", "switch on", "activate", "enable", "start", "beep")) and not any(x in t for x in ("turn off", "switch off", "deactivate", "disable", "stop")):
            ok, detail = esp32_request("/buzzer", {"state": "on"})
            return "Buzzer is on." if ok else detail
        if any(x in t for x in ("turn off", "switch off", "deactivate", "disable", "stop", "silence")):
            ok, detail = esp32_request("/buzzer", {"state": "off"})
            return "Buzzer is off." if ok else detail

    return None


# ---------------- GROQ ----------------
def ask_groq(user_text):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Your Groq API key isn't set. Add it to the GROQ_API_KEY environment variable and restart SAM-JARVIS."
    if Groq is None:
        return "The Groq package isn't installed. Run: py -m pip install groq"
    try:
        client = Groq(api_key=api_key)
        with history_lock:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.extend(chat_history[-12:])
            messages.append({"role": "user", "content": user_text})
        result = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=700,
        )
        reply = result.choices[0].message.content or "I couldn't generate a reply."
        reply = re.sub(r"\bboss\b", USER_NAME, reply, flags=re.IGNORECASE)
        with history_lock:
            chat_history.append({"role": "user", "content": user_text})
            chat_history.append({"role": "assistant", "content": reply})
            del chat_history[:-24]
        return reply
    except Exception as exc:
        print("Groq error:", repr(exc))
        return "I couldn't contact Groq. Check your API key, internet connection, and model name."


def process_command(text, speak_reply=True):
    text = (text or "").strip()
    if not text:
        return "I didn't catch that. Please try again."
    set_status(last_heard=text, mode="Processing", message=f"You said: {text}")
    reply = hardware_command(text)
    if reply is None:
        reply = ask_groq(text)
    set_status(mode="Speaking", message=reply, last_reply=reply)
    if speak_reply:
        speak(reply)
    return reply


# ---------------- WAKE WORD LOOP ----------------
def wake_listener():
    global audio_stream
    if Model is None:
        set_status(mode="Error", message="openWakeWord is missing. Install it with pip.")
        return

    audio_queue = queue.Queue(maxsize=80)

    def audio_callback(indata, frames, time_info, status):
        if status:
            print("Audio status:", status)
        try:
            audio_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    try:
        # openWakeWord downloads/loads its default model files on first use.
        wake_model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
        set_status(mode="Standby", message='Listening locally for "Hey Jarvis".')
        print('SAM-JARVIS is listening for "Hey Jarvis". Press Ctrl+C to stop.')

        while not wake_stop_event.is_set():
            try:
                with sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="int16",
                    blocksize=FRAME_SIZE,
                    callback=audio_callback,
                ) as audio_stream:
                    buffer = np.empty(0, dtype=np.int16)
                    while not wake_stop_event.is_set():
                        try:
                            chunk = audio_queue.get(timeout=0.5)
                        except queue.Empty:
                            continue
                        buffer = np.concatenate((buffer, chunk.astype(np.int16, copy=False)))
                        while len(buffer) >= FRAME_SIZE:
                            frame = buffer[:FRAME_SIZE]
                            buffer = buffer[FRAME_SIZE:]
                            prediction = wake_model.predict(frame)
                            score = max(prediction.values()) if prediction else 0.0
                            if score >= WAKE_THRESHOLD:
                                # Leave the stream context before starting command recognition.
                                raise WakeDetected()
            except WakeDetected:
                # Stream closes here so SpeechRecognition can use the microphone.
                pass

            if wake_stop_event.is_set():
                break

            set_status(mode="Wake detected", message="Yes, Sambit? I'm listening.")
            speak(f"Yes, {USER_NAME}?")
            recognizer = sr.Recognizer()
            try:
                with sr.Microphone(sample_rate=SAMPLE_RATE) as source:
                    set_status(mode="Listening for command", message="Say your command now.")
                    recognizer.adjust_for_ambient_noise(source, duration=0.35)
                    audio = recognizer.listen(
                        source,
                        timeout=COMMAND_TIMEOUT,
                        phrase_time_limit=PHRASE_LIMIT
                    )
                set_status(mode="Recognizing", message="Transcribing your command…")
                command = recognizer.recognize_google(audio, language="en-IN")
                process_command(command, speak_reply=True)
            except sr.WaitTimeoutError:
                speak("I didn't hear a command, Sambit.")
                set_status(mode="Standby", message='Listening for "Hey Jarvis".')
            except sr.UnknownValueError:
                speak("Sorry, Sambit, I couldn't understand that.")
                set_status(mode="Standby", message='Listening for "Hey Jarvis".')
            except sr.RequestError:
                speak("Speech recognition needs an internet connection.")
                set_status(mode="Standby", message='Listening for "Hey Jarvis".')
            except Exception as exc:
                print("Command listening error:", repr(exc))
                set_status(mode="Standby", message='Listening for "Hey Jarvis".')

            time.sleep(WAKE_COOLDOWN_SECONDS)
            set_status(mode="Standby", message='Listening locally for "Hey Jarvis".')

    except Exception as exc:
        print("Wake listener error:", repr(exc))
        set_status(mode="Error", message=f"Wake-word listener error: {exc}")
    finally:
        audio_stream = None


class WakeDetected(Exception):
    pass


def start_wake_listener():
    global wake_thread
    if wake_thread and wake_thread.is_alive():
        return
    wake_stop_event.clear()
    wake_thread = threading.Thread(target=wake_listener, daemon=True)
    wake_thread.start()


# ---------------- WEB DASHBOARD ----------------
HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SAM-JARVIS</title>
<style>
:root{color-scheme:dark;--bg:#07111f;--panel:#101f33;--line:#233a55;--accent:#35d5ff;--text:#eaf5ff;--muted:#9bb0c7}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top,#12304b,#07111f 55%);font-family:Segoe UI,Arial,sans-serif;color:var(--text);min-height:100vh}
header{padding:22px 5%;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center;gap:15px;flex-wrap:wrap}
h1{margin:0;font-size:clamp(22px,4vw,32px);letter-spacing:2px}h1 span{color:var(--accent)}.sub{color:var(--muted);margin-top:6px}
main{max-width:1050px;margin:26px auto;padding:0 16px;display:grid;grid-template-columns: minmax(0,1.5fr) minmax(260px,1fr);gap:18px}
.panel{background:rgba(16,31,51,.94);border:1px solid var(--line);border-radius:18px;padding:20px;box-shadow:0 15px 45px #0003}
.status{display:flex;align-items:center;gap:10px;color:var(--accent);font-weight:600}.dot{width:10px;height:10px;border-radius:50%;background:#39e58c;box-shadow:0 0 14px #39e58c}
#statusMsg{color:var(--muted);min-height:40px;margin:12px 0}.chat{height:340px;overflow:auto;display:flex;flex-direction:column;gap:10px;padding:10px 0}
.msg{padding:12px 14px;border-radius:14px;max-width:90%;white-space:pre-wrap;line-height:1.45}.user{align-self:flex-end;background:#12486a}.assistant{align-self:flex-start;background:#1b2e45}
form{display:flex;gap:8px;margin-top:12px}input{min-width:0;flex:1;background:#081727;border:1px solid var(--line);border-radius:12px;padding:13px;color:white;font-size:16px}
button{border:0;border-radius:12px;padding:12px 16px;background:var(--accent);color:#03101a;font-weight:700;cursor:pointer}button.secondary{background:#203852;color:var(--text);border:1px solid var(--line)}.controls{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:18px}.controls button{min-height:52px}.wide{grid-column:1/-1}.hint{font-size:13px;color:var(--muted);line-height:1.5;margin-top:16px}
@media(max-width:760px){main{grid-template-columns:1fr}.chat{height:280px}}
</style>
</head>
<body>
<header><div><h1><span>SAM</span>-JARVIS</h1><div class="sub">Personal AI • ESP32 Smart Control</div></div><div class="status"><span class="dot"></span><span id="mode">Connecting…</span></div></header>
<main>
<section class="panel"><h2>Chat with Jarvis</h2><div id="statusMsg">Connecting to assistant…</div><div class="chat" id="chat"><div class="msg assistant">Hey Sambit! Say “Hey Jarvis” to wake me up, or type below.</div></div>
<form id="chatForm"><input id="prompt" placeholder="Ask Jarvis something…" autocomplete="off"><button>Send</button></form>
<div class="hint">Wake word runs locally on your PC. The typed chat uses your Groq API key on the PC, not in this page.</div></section>
<aside class="panel"><h2>ESP32 Controls</h2><div class="sub">Device: 192.168.1.10</div><div class="controls">
<button onclick="hardware('turn on the LED')">LED ON</button><button class="secondary" onclick="hardware('turn off the LED')">LED OFF</button>
<button onclick="hardware('turn on the buzzer')">BUZZER ON</button><button class="secondary" onclick="hardware('turn off the buzzer')">BUZZER OFF</button>
<button class="wide secondary" onclick="hardware('turn everything off')">ALL OFF</button>
<button class="wide secondary" onclick="fetch('/api/wake',{method:'POST'}).then(refreshStatus)">Restart wake listener</button>
</div><div class="hint">If a control doesn't respond, check that the ESP32 and PC are on the same Wi-Fi network and that the ESP32 IP address is correct.</div></aside>
</main>
<script>
const chat=document.getElementById('chat');
function addMsg(role,text){const d=document.createElement('div');d.className='msg '+role;d.textContent=text;chat.appendChild(d);chat.scrollTop=chat.scrollHeight}
async function sendText(text){if(!text.trim())return;addMsg('user',text);document.getElementById('prompt').value='';try{const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text})});const j=await r.json();addMsg('assistant',j.reply||j.error||'No reply');}catch(e){addMsg('assistant','Could not reach SAM-JARVIS. Is app.py running?')}}
document.getElementById('chatForm').addEventListener('submit',e=>{e.preventDefault();sendText(document.getElementById('prompt').value)});
async function hardware(text){await sendText(text)}
async function refreshStatus(){try{const r=await fetch('/api/status');const s=await r.json();document.getElementById('mode').textContent=s.mode;document.getElementById('statusMsg').textContent=s.message;}catch(e){}}
setInterval(refreshStatus,1500);refreshStatus();
</script>
</body></html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/status")
def api_status():
    return jsonify(get_status())


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify({"error": "Please enter a message."}), 400
    reply = process_command(message, speak_reply=False)
    return jsonify({"reply": reply})


@app.route("/api/wake", methods=["POST"])
def api_wake():
    start_wake_listener()
    return jsonify({"ok": True, "message": "Wake listener is running or starting."})


if __name__ == "__main__":
    print("=" * 48)
    print("SAM-JARVIS 2.0")
    print("User name: Sambit (never Boss)")
    print(f"ESP32: {ESP32_BASE_URL}")
    print(f"Dashboard: http://{HOST}:{PORT}")
    print("=" * 48)
    start_wake_listener()
    # Disable Flask reloader so the wake-word thread isn't started twice.
    app.run(host=HOST, port=PORT, debug=False, use_reloader=False, threaded=True)
