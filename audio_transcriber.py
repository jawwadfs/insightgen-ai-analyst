"""
Audio Transcription Module for InsightGen.
Supports Whisper (SpeechRecognition/OpenAI Whisper) and Gemini Multimodal Audio.
"""

import io
import warnings
warnings.filterwarnings("ignore")
from typing import Tuple, Optional


def transcribe_audio_gemini(audio_bytes: bytes, api_key: str) -> Tuple[bool, str]:
    """Transcribes audio using Gemini 1.5 Flash's multimodal audio comprehension."""
    try:
        import google.generativeai as genai
        clean_key = str(api_key).strip().strip('"').strip("'")
        genai.configure(api_key=clean_key)
        
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "You are an audio transcription engine for a data analytics platform. "
            "Listen to this audio clip and transcribe the spoken business or analytics question word for word. "
            "Output ONLY the transcribed text. Do not add quotes, punctuation explanations, or extra commentary."
        )
        
        response = model.generate_content([
            prompt,
            {"mime_type": "audio/wav", "data": audio_bytes}
        ])
        
        if response and response.text:
            cleaned_text = response.text.strip().strip('"').strip("'")
            return True, cleaned_text
        return False, "Empty transcription returned from model."
    except Exception as e:
        return False, f"Gemini Audio Transcription Error: {str(e)}"


def transcribe_audio_whisper(audio_bytes: bytes) -> Tuple[bool, str]:
    """Transcribes audio locally using SpeechRecognition with Whisper or Google Speech fallback."""
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        
        # Wrap bytes in a BytesIO stream
        audio_stream = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_stream) as source:
            audio_data = r.record(source)
            
        # Try local whisper first if available, else recognize_google
        try:
            text = r.recognize_whisper(audio_data, model="base")
            return True, text.strip()
        except Exception:
            # Fallback to speech recognition
            try:
                text = r.recognize_google(audio_data)
                return True, text.strip()
            except Exception as e_google:
                return False, f"Local Speech Recognition Error: {str(e_google)}"
    except Exception as e:
        return False, f"Audio Decoding Error: {str(e)}"


def transcribe_user_speech(audio_bytes: bytes, api_key: Optional[str] = None) -> Tuple[bool, str]:
    """
    Unified speech-to-text pipeline:
    Tries Gemini Multimodal Audio first if API key is provided,
    otherwise falls back to local Whisper/SpeechRecognition.
    """
    if not audio_bytes or len(audio_bytes) < 1000:
        return False, "Audio recording was too short or empty."

    if api_key and len(str(api_key).strip()) > 10:
        success, res = transcribe_audio_gemini(audio_bytes, api_key)
        if success and len(res.strip()) > 0:
            return True, res

    # Fallback to Whisper / local engine
    return transcribe_audio_whisper(audio_bytes)
