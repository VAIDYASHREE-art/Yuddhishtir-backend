import os
import sys
import json
import csv
import streamlit as st
import whisper
import openai
import librosa

from yuddhishitir import YuddhishtirRAGSystem, build_rag_prompt
from yuddhishitir.generate_response import generate_response
from yuddhishitir.tts_handler import TTSWrapper


# ───────────────────────────────────────────
# App Config
# ───────────────────────────────────────────
st.set_page_config(page_title="AIIMS Assistant", page_icon="🩺")
st.title("🧠 AIIMS Voice Assistant")
st.write("Your calm companion for voice + AI-powered hospital help.")

# ───────────────────────────────────────────
# Session Memory
# ───────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ───────────────────────────────────────────
# Mic-recording check
# ───────────────────────────────────────────
try:
    import sounddevice as sd
    from scipy.io.wavfile import write
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

# ───────────────────────────────────────────
# Path & API Config
# ───────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

openai.api_key = st.secrets["openai"]["api_key"]
whisper_model = whisper.load_model("base")
tts = TTSWrapper()

# ───────────────────────────────────────────
# Utils
# ───────────────────────────────────────────
def reduce_noise_from_file(input_path: str, sr: int = 16000) -> str:
    y, _ = librosa.load(input_path, sr=sr)
    cleaned_path = "cleaned_audio.wav"
    librosa.output.write_wav(cleaned_path, y, sr)
    return cleaned_path

def transcribe_audio(file_path: str) -> str:
    result = whisper_model.transcribe(file_path)
    return result.get("text", "").strip()

def log_conversation(user_input, assistant_reply):
    os.makedirs("conversations", exist_ok=True)
    with open("conversations/log.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([user_input, assistant_reply])

# ───────────────────────────────────────────
# UI Input Mode Switch
# ───────────────────────────────────────────
mode = st.radio("Choose input mode", ["Text", "Voice"])

if mode == "Text":
    prompt = st.text_input("Ask a medical treatment or procedure cost")
    if st.button("Send"):
        if not prompt:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                answer = generate_response(prompt)
                st.session_state.chat_history.append(("You", prompt))
                st.session_state.chat_history.append(("AIIMS Assistant", answer))
                log_conversation(prompt, answer)
                tts.speak(answer)

elif mode == "Voice":
    st.write("🎤 Record with mic or upload a WAV/MP3 file")

    if not SOUNDDEVICE_AVAILABLE:
        st.warning("Mic recording disabled (requires `sounddevice`, `scipy`).")

    col1, col2 = st.columns(2)
    with col1:
        record = st.button("🔴 Record 5 sec") if SOUNDDEVICE_AVAILABLE else False
    with col2:
        uploaded = st.file_uploader("Or upload audio", type=["wav", "mp3"])

    raw_path = None
    if record:
        fs, dur = 16000, 5
        st.info("Recording...")
        rec = sd.rec(int(dur * fs), samplerate=fs, channels=1)
        sd.wait()
        raw_path = "raw_audio.wav"
        write(raw_path, fs, rec)
        st.success("Recording done.")

    if uploaded:
        raw_path = "raw_audio.wav"
        with open(raw_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success("File uploaded.")

    if raw_path and os.path.exists(raw_path):
        st.audio(raw_path, format="audio/wav")
        clean_path = reduce_noise_from_file(raw_path)
        st.info("Transcribing…")
        text = transcribe_audio(clean_path)
        st.markdown(f"**You said:**  \n{text}")
        st.info("Generating response…")
        answer = generate_response(text)
        st.markdown(f"**Yuddhishtir says:**  \n{answer}")
        log_conversation(text, answer)
        tts.speak(answer)

# ───────────────────────────────────────────
# Chat History
# ───────────────────────────────────────────
st.markdown("---")
for sender, msg in st.session_state.chat_history[::-1]:
    st.markdown(f"**{sender}:** {msg}")

# ───────────────────────────────────────────
# Reset
# ───────────────────────────────────────────
if st.button("Reset Session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()