print("✅ Backend booting up... importing modules...")
import os
import tempfile
import librosa
import uvicorn
import whisper
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from yuddhishitir.generate_response import generate_response
from yuddhishitir.tts_handler import TTSWrapper

app = FastAPI()
whisper_model = whisper.load_model("base")
tts = TTSWrapper()

# ✅ CORS setup for Render or localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with exact frontend URL in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        y, sr = librosa.load(tmp_path, sr=16000)
        cleaned_path = tmp_path.replace(".wav", "_clean.wav")
        librosa.output.write_wav(cleaned_path, y, sr)

        result = whisper_model.transcribe(cleaned_path)
        return {"transcription": result.get("text", "").strip()}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/ask")
async def ask_text(body: TextInput):
    try:
        answer = generate_response(body.text)
        return {"response": answer}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/speak")
async def speak_text(body: TextInput):
    try:
        tts.speak(body.text)
        return FileResponse("output.wav", media_type="audio/wav", filename="output.wav")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)