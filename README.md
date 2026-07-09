# Rudimentary Voice AI Chatbot

Voice pipeline built with [Pipecat](https://github.com/pipecat-ai/pipecat):

mic (WebRTC or local) -> Silero VAD -> Groq (Whisper STT, bilingual English/Urdu) -> Groq (Llama 3.3 70B) -> Kokoro (local TTS) -> speakers

## Setup

1. Create a `.env` file with your Groq API key:

   ```
   GROQ_API_KEY=your_key_here
   # Optional: override the default Kokoro voice
   # KOKORO_VOICE=af_heart
   ```

   Get a free key at https://console.groq.com. No other API keys are needed — TTS runs locally via Kokoro.

2. Install dependencies (venv lives on `D:\venvs\pipecat-voice` — C: was nearly full):

   ```
   D:\venvs\pipecat-voice\Scripts\python.exe -m pip install "pipecat-ai[groq,local,silero]" python-dotenv kokoro-onnx
   ```

## Run

Two modes:

```
D:\venvs\pipecat-voice\Scripts\python.exe bot.py             # WebRTC server; open http://localhost:7860
D:\venvs\pipecat-voice\Scripts\python.exe bot.py --local     # Talk directly via the local mic/speakers
```

Talk into your microphone; the bot replies out loud. Ctrl+C to stop.

On first run, Kokoro downloads its model files (~87 MB) to `~/.cache/pipecat/kokoro-onnx/`. This only happens once.

## Notes

- `--local` uses `LocalAudioTransport` (your PC's mic/speakers directly via PyAudio, no browser). The default WebRTC mode serves a browser UI via `pipecat.runner.run`.
- STT (`BilingualGroqSTTService`) transcribes each utterance twice concurrently — once forced to English, once forced to Urdu — and keeps whichever result has higher confidence. This correctly handles both languages without Whisper's broader auto-detect misfiring into a third language (it previously misdetected Urdu as Chinese). The system prompt still forces the LLM to always reply in English. See `EDGE_CASES.md` (section B1) for the full story.
- Kokoro TTS does not support Urdu script — if the LLM ever responds in Urdu, synthesis will fail. This is a known gap, documented in `EDGE_CASES.md`.
- Swap `KOKORO_VOICE` in `.env` to change the TTS voice (e.g. `am_michael`, `bf_emma`, `bm_george`).
- Set `LOG_LEVEL=DEBUG` in the environment to see per-turn diagnostics (raw STT transcripts, etc.) in addition to the default `CONVO`-level transcript log.
- See `EDGE_CASES.md` for the full edge-case testing log and known issues.
