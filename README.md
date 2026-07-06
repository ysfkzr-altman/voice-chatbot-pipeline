# Rudimentary Voice AI Chatbot

Local voice pipeline built with [Pipecat](https://github.com/pipecat-ai/pipecat):

mic -> Silero VAD -> Groq (Whisper STT) -> Gemini Flash (LLM) -> Cartesia Sonic (TTS) -> speakers

## Setup

1. Copy `.env.example` to `.env` and fill in your API keys:
   - `GROQ_API_KEY` — https://console.groq.com
   - `GOOGLE_API_KEY` — Gemini API key from your Google AI Studio / Gemini account
   - `CARTESIA_API_KEY` — https://cartesia.ai

2. Install dependencies (venv lives on `D:\venvs\pipecat-voice` — C: was nearly full):

   ```
   D:\venvs\pipecat-voice\Scripts\python.exe -m pip install "pipecat-ai[google,groq,local,silero]" python-dotenv
   ```

## Run

```
D:\venvs\pipecat-voice\Scripts\python.exe bot.py
```

Talk into your microphone; the bot replies out loud. Ctrl+C to stop.

## Notes

- This uses `LocalAudioTransport` (your PC's mic/speakers directly via PyAudio) — no browser or telephony transport involved, which is why this is "rudimentary."
- Swap `gemini-2.5-flash` in `bot.py` for another Gemini model if desired.
- Swap `CARTESIA_VOICE_ID` in `.env` to change the TTS voice.
