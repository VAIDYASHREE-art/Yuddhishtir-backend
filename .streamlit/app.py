import os
import sys
import csv
import streamlit as st
import whisper
import openai
import librosa
from yuddhishitir.generate_response import generate_response
from yuddhishitir.tts_handler import TTSWrapper

# ───────────────────────────
# Config & Setup
# ───────────────────────────
st.set_page_config(page_title="AIIMS Assistant", page_icon="🩺")
st.markdown("<h2 style='text-align:center;'>🧠 AIIMS Voice Assistant</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Your calm companion for voice + AI-powered hospital help.</p>", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

try:
    import sounddevice as sd
    from scipy.io.wavfile import write
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

openai.api_key = st.secrets["openai"]["api_key"]
whisper_model = whisper.load_model("base")
tts = TTSWrapper()

# ───────────────────────────
# Utility Functions
# ───────────────────────────
def reduce_noise(input_path: str, sr: int = 16000) -> str:
    y, _ = librosa.load(input_path, sr=sr)
    cleaned_path = "cleaned_audio.wav"
    librosa.output.write_wav(cleaned_path, y, sr)
    return cleaned_path

def transcribe_audio(file_path: str) -> str:
    if os.path.getsize(file_path) < 1000:
        return "[Empty or invalid audio]"
    result = whisper_model.transcribe(file_path)
    return result.get("text", "").strip()

def log_convo(user_msg, reply):
    os.makedirs("conversations", exist_ok=True)
    with open("conversations/log.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([user_msg, reply])

# ───────────────────────────
# Frontend Layout
# ───────────────────────────
st.markdown("### 🔄 Select Input Mode")
mode = st.radio("", ["💬 Text", "🎙️ Voice"], horizontal=True)

if mode == "💬 Text":
    col1, col2 = st.columns([4, 1])
    with col1:
        prompt = st.text_input("Enter your medical query")
    with col2:
        send = st.button("📨 Send")
    if send and prompt:
        with st.spinner("🤖 Generating response..."):
            reply = generate_response(prompt)
            st.session_state.chat_history.append(("You", prompt))
            st.session_state.chat_history.append(("AIIMS Assistant", reply))
            log_convo(prompt, reply)
            tts.speak(reply)
            st.success("✅ Done!")

elif mode == "🎙️ Voice":
    st.markdown("#### 🎧 Record or Upload")
    col1, col2 = st.columns(2)
    record = col1.button("🎙️ Tap to Record (5s)") if SOUNDDEVICE_AVAILABLE else False
    uploaded = col2.file_uploader("📂 Upload Audio", type=["wav", "mp3"])

    raw_path = None
    if record:
        fs = 16000
        st.info("Recording...")
        rec = sd.rec(int(5 * fs), samplerate=fs, channels=1)
        sd.wait()
        raw_path = "raw_audio.wav"
        write(raw_path, fs, rec)
        st.success("🎙️ Recorded.")

    elif uploaded:
        raw_path = "raw_audio.wav"
        with open(raw_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success("📂 File uploaded.")

    if raw_path and os.path.exists(raw_path):
        st.audio(raw_path)
        with st.spinner("🔄 Cleaning & transcribing..."):
            clean = reduce_noise(raw_path)
            text = transcribe_audio(clean)
        st.markdown(f"**🗣️ You said:** {text}")
        with st.spinner("🤖 Thinking..."):
            reply = generate_response(text)
            st.markdown(f"**💡 Yuddhishtir says:** {reply}")
            log_convo(text, reply)
            tts.speak(reply)

# ───────────────────────────
# Chat History + Reset
# ───────────────────────────
st.markdown("---")
with st.expander("🕓 Chat History"):
    for sender, msg in st.session_state.chat_history[::-1]:
        st.markdown(f"**{sender}:** {msg}")

if st.button("🔁 Reset Session"):
    st.session_state.chat_history.clear()
    st.experimental_rerun()