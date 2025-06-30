# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 17:18:31 2025

@author: DELL
"""

# main.py


from yuddhishitir.generate_response import generate_response
from yuddhishitir.tts_handler import TTSWrapper
# Initialize TTS once
tts = TTSWrapper()
tts.speak("Welcome to AIIMS assistant! How can I help you today?", speaker="en_1", language="en")

# Sample usage
query = "What are the timings for the cardiology OPD?"
response = generate_response(query)
print("Yuddhishtir says:", response)

tts.speak(response)

if __name__ == "__main__":
    while True:
        query = input("\n🗣️  Your question: ")
        if query.lower() in ['exit', 'quit']:
            print("👋 Exiting Yuddhishtir.")
            break
        response = generate_response(query)
        print(f"\n🤖 Yuddhishtir says:\n{response}")