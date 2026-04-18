
from typing import Dict
import requests
import streamlit as st

TRANSCRIBE_URL = st.secrets["MODAL_TRANSCRIBE_URL"]
CHAT_URL = st.secrets["MODAL_CHAT_URL"]


def transcribe_audio(audio_bytes: bytes) -> Dict:
    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
    resp = requests.post(TRANSCRIBE_URL, files=files, timeout=120)
    resp.raise_for_status()
    return resp.json()


def chat_with_bot(message: str, mode: str) -> Dict:
    payload = {"message": message, "mode": mode}
    resp = requests.post(CHAT_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()
