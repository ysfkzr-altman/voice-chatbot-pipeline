# Voice-In, Text-Out AI Chatbot

Pipeline built with [Pipecat](https://github.com/pipecat-ai/pipecat):

mic (WebRTC or local) -> Silero VAD -> Groq (Whisper STT, bilingual English/Urdu) -> Groq (Llama 3.3 70B) -> reply delivered as text (RTVI message or console) — **no TTS, no speaker output**.

This is a sibling variant of the voice-in/voice-out bot on the `fix-edge-cases` branch: same input pipeline (mic, VAD, bilingual STT), but replies are delivered as text instead of being spoken. It also includes a worked example of LLM tool-calling against a real website (`honda.com.pk`), not fake/hardcoded data.

## Setup

1. Create a `.env` file with your Groq API key:

   ```
   GROQ_API_KEY=your_key_here
   ```

   Get a free key at https://console.groq.com. No other API keys are needed.

2. Install dependencies (venv lives on `D:\venvs\pipecat-voice` — C: was nearly full):

   ```
   D:\venvs\pipecat-voice\Scripts\python.exe -m pip install "pipecat-ai[groq,local,silero]" python-dotenv beautifulsoup4 curl_cffi
   ```

## Run

Two modes:

```
D:\venvs\pipecat-voice\Scripts\python.exe bot.py             # WebRTC server; open http://localhost:7860
D:\venvs\pipecat-voice\Scripts\python.exe bot.py --local     # Talk directly via the local mic
```

Speak into your microphone; the bot's reply is delivered as text — over WebRTC it arrives at the connected client as an RTVI `bot-llm-text` message, in `--local` mode it only shows up in the console (`CONVO`-level log line), since there's no client to display it to. Ctrl+C to stop.

Set `LOG_LEVEL=CONVO` to see just the conversation transcript (`User:`/`Bot:` lines) instead of full debug output — much cleaner for `--local` mode. Note this only applies to `--local`: the WebRTC path resets logging back to full debug internally right before starting the server (a `pipecat.runner.run.main()` behavior, not something this file controls).

## Tool-calling demo

The bot can answer real questions about Honda Pakistan by fetching live data, not guessing:

- `check_honda_price` — real, current starting prices from honda.com.pk's homepage mega-menu (e.g. "How much does the Civic cost?")
- `browse_honda_page` — fetches and reads any of a known set of real pages on the site (model specs, dealer/contact info, promotions, company info, policies) for open-ended questions

Both tools fall back to a recently-cached result if a live fetch fails, and distinguish "the site itself is unreachable/broken" from "this specific model/page doesn't exist" — a naive scrape failure (e.g. a bot-challenge page from Cloudflare) used to silently report every model as "not found" with no signal anything was actually wrong.

## Testing

`evals/` holds automated scenarios (synthesized speech via Kokoro, no human needed, no TTS output needed to run them since this pipeline doesn't use it). Run the whole suite:

```
scripts/run_all_evals.sh
```

Starts a fresh `bot.py` process per scenario, runs it, tears it down, and prints a pass/fail summary. Pass one or more names to run a subset: `scripts/run_all_evals.sh backchannel_test tool_calling_test`. Bot/eval logs land in `eval_logs/` (gitignored).

Some scenarios use a local Ollama judge (`eval:` semantic checks) — if Ollama isn't running, those fail with an `APIConnectionError` unrelated to the bot itself; the summary flags which scenarios that applies to.

## What's different from the voice variant (`fix-edge-cases` branch)

This branch ported everything from the voice variant's hardening work that **isn't** specific to the STT/audio-input layer: startup self-check, real circuit breaker with exponential backoff, context summarization for long conversations, interrupted-response context handling, missing-key/port-conflict fast-fails, and an ambiguous-input clarifying-question instruction.

Deliberately **not** ported — these are audio-input/STT-layer concerns, out of scope for this variant:
- The backchannel filter (dropping "Mm-hmm"-style filler before it reaches the LLM)
- Smart Turn / VAD calibration tuning
- The Whisper-hallucination phrase filter
- The Urdu-misclassification confidence bias tuning (`SHORT_UTTERANCE_EN_BIAS`)

Practical consequence: this bot can still occasionally mistranscribe speech as Urdu script gibberish (especially speech with natural pauses/filler words — real human speech, not the clean synthesized audio the eval suite tests with), and that garbled input can in turn confuse the LLM's tool-calling into a malformed function call that Groq's API rejects outright. When that happens, the circuit breaker/fallback message correctly catches it and asks you to repeat yourself — the bot doesn't crash or hang, but the underlying transcription issue isn't fixed here. See `fix-edge-cases` for the full audio-pipeline hardening, including that fix.

## Notes

- `--local` uses `LocalAudioTransport` (your PC's mic directly via PyAudio, no browser).
- STT (`BilingualGroqSTTService`) transcribes each utterance twice concurrently — once forced to English, once forced to Urdu — and keeps whichever result has higher confidence. Unlike the voice variant, there's no technical reason this bot couldn't reply in Urdu too (no TTS to crash) — the system prompt still forces English-only replies, kept as-is for behavioral consistency between the two variants.
- STT model is `whisper-large-v3-turbo`, not `whisper-large-v3` — measured directly in this pipeline: ~6-10x faster (3s → 0.3-0.5s) with no observed accuracy regression in testing, though Whisper's turbo variants are documented to trade a little accuracy for speed on less-common languages/accents.
- A `TextSanitizer` strips any raw HTML-like tags from LLM output before delivery — defensive (no observed bug), since this variant's system prompt uniquely allows rich formatting and a client that renders markdown-to-HTML without sanitizing would be at risk from any stray tag the LLM echoes back.
- See `EDGE_CASES.md` for the original edge-case audit this whole project started from — note its header explains most of it is about the *voice* variant specifically.
