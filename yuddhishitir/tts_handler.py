import os
import random
import re
import torch
import pickle
from TTS.api import TTS
from torch.serialization import add_safe_globals

# Optional: preload known safe classes (for PyTorch >=2.6)
known_safe_classes = [
    "TTS.tts.configs.xtts_config.XttsConfig",
    "TTS.tts.models.xtts.XttsAudioConfig",
    "TTS.config.shared_configs.BaseDatasetConfig",
    "TTS.tts.models.xtts.Xtts",
    "TTS.tts.models.xtts.XttsModelConfig",
]

for class_path in known_safe_classes:
    try:
        mod_path, class_name = class_path.rsplit(".", 1)
        mod = __import__(mod_path, fromlist=[class_name])
        klass = getattr(mod, class_name)
        add_safe_globals([klass])
    except Exception as e:
        print(f"⚠️ Could not register {class_path}: {e}")

class TTSWrapper:
    def __init__(self):
        try:
            self.tts = TTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=False,
                gpu=False
            )
        except pickle.UnpicklingError as e:
            missing = re.findall(r"TTS\.[\w\.]+", str(e))
            for class_path in missing:
                try:
                    mod_path, class_name = class_path.rsplit(".", 1)
                    mod = __import__(mod_path, fromlist=[class_name])
                    klass = getattr(mod, class_name)
                    add_safe_globals([klass])
                    print(f"✅ Registered dynamically: {class_path}")
                except Exception as inner_e:
                    print(f"⚠️ Could not dynamically register {class_path}: {inner_e}")
            # Retry after dynamic registration
            self.tts = TTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=False,
                gpu=False
            )

    def speak(self, text, speaker=None, speaker_wav_path=None, language="en"):
     available_speakers = getattr(self.tts, "speakers", [])
    
    # If no speaker selected and list is non-empty, pick randomly
     if not speaker:
         if available_speakers:
             speaker = random.choice(available_speakers)
             print(f"🎛️ Using speaker: {speaker}")
         elif speaker_wav_path:
             speaker = None  # XTTS will extract embedding from wav
             print("🎛️ Using provided reference audio.")
         else:
             raise ValueError("❌ No valid speaker ID or reference audio provided.")

     output_path = "output.wav"
     self.tts.tts_to_file(
         text=text,
         speaker=speaker,
         speaker_wav=speaker_wav_path,
         language=language,
         file_path=output_path
     )
     print(f"✅ Speech saved to {output_path}")