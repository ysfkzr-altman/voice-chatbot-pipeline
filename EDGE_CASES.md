# Edge Case Testing — Voice AI Pipeline

> **Note:** this is the original edge-case audit from early in the project,
> kept as-is for historical context - it predates most of the fixes on this
> branch and its per-item status markers weren't updated as those landed
> (e.g. A4's barge-in note below says "not yet re-tested with Kokoro TTS",
> but it has been, extensively - see `evals/bargein_test.yaml`). The actively
> maintained list of confirmed problems and their fixes is
> `Pending_Edge_Cases_and_Solutions.pdf` plus the `Confirmed problem #N`
> comments throughout `bot.py` itself; `README.md`'s Testing section has the
> current eval suite. Treat this file as "what we originally found," not
> "what's still broken."

## Summary

Comprehensive edge case catalog for the `bot.py` voice AI pipeline (Groq Whisper STT → Groq LLM Llama 3.3 70B → Kokoro TTS, WebRTC or `--local` PyAudio). Cases are grouped by pipeline layer. Each entry notes its status: **CONFIRMED** (observed in testing), **EXPECTED** (not yet tested, behavior predicted from architecture), or **OPEN** (known issue, unresolved). See the plan file for the full testing methodology.

---

## A. Audio / Mic Input

### A1. Background noise triggering false VAD turns
- **Description**: Ambient sounds (TV, music, traffic, AC hum) exceed the VAD threshold and are treated as user speech. STT receives noise audio, produces a garbled or empty transcript, and the LLM responds to nothing — the user hears a confusing non-sequitur.
- **Test method**: Play music near the mic in `--local` mode; observe whether the pipeline fires a turn.
- **Status**: EXPECTED — not yet tested. SileroVADAnalyzer has configurable thresholds; may require tuning for noisy environments.

### A2. Echo feedback (speakerphone mode)
- **Description**: Bot's own TTS output is picked up by the mic, transcribed, and fed back into the LLM — the bot "hears itself" and begins responding to its own words, potentially looping.
- **Test method**: Remove headphones, use laptop speakers and mic simultaneously. Listen for self-response loops.
- **Status**: EXPECTED — wired headphones (current setup) prevent this. Speakerphone mode is untested.

### A3. Very quiet speech / whispering
- **Description**: Whispered speech may fall below SileroVAD's activation threshold and never trigger a user turn. The pipeline waits indefinitely.
- **Test method**: Whisper a question at the mic; check whether a `TranscriptionFrame` is produced.
- **Status**: EXPECTED — not tested.

### A4. Barge-in / interruption while bot is speaking
- **Description**: User speaks while the bot's TTS audio is playing. The expected behavior is that VAD detects the user's voice, broadcasts an interruption signal, and the bot stops speaking mid-response.
- **Test method**: Run `python bot.py --local`, trigger a long bot response, speak over it. Check DEBUG logs for `"User started speaking"` during active TTS playback windows.
- **Status**: OPEN — VAD and interruption broadcast confirmed wired correctly by pipecat default. Live testing was blocked by a Cartesia billing issue (HTTP 402) before the DirectSound input device fix could be re-tested. Not yet re-tested with Kokoro TTS.
- **Note**: Barge-in behavior may differ between `--local` (raw PyAudio) and WebRTC (browser audio stack with its own AEC). Test both when possible.

### A5. Mid-sentence pause causing early turn cutoff
- **Description**: User pauses mid-thought (e.g., "I want to ask about... um... the deadline"). SileroVAD interprets the silence as end-of-turn and fires. The partial sentence is transcribed and sent to the LLM, which responds to an incomplete question.
- **Test method**: Speak a sentence with a deliberate 1-2 second pause in the middle; observe what gets transcribed.
- **Status**: EXPECTED — VAD silence threshold is tunable via `SileroVADAnalyzer` params. Default behavior not yet characterized.

### A6. Non-speech sounds as false turns
- **Description**: Coughing, sneezing, laughing, or clicking near the mic may cross the VAD threshold and produce a short noise-only audio segment. Whisper may transcribe this as a real word ("hmm", "yeah") or hallucinate something entirely.
- **Test method**: Cough directly into the mic; observe transcript output.
- **Status**: EXPECTED — Whisper hallucination on short non-speech audio is a known documented behavior in OpenAI's Whisper research.

### A7. Rapid back-to-back turns
- **Description**: User speaks again immediately after finishing their first turn, before the LLM has responded. Depending on pipeline state, the second turn may be queued, dropped, or collide with the in-progress LLM generation.
- **Test method**: Ask two questions in quick succession with minimal gap.
- **Status**: EXPECTED — not tested.

---

## B. Language & Transcription

### B1. Non-English input misdetected as wrong language
- **Description**: When Whisper's `language` parameter is set to `"en"`, non-English speech is force-fit into English phonemes, producing garbled output. With unconstrained auto-detection, short or ambiguous audio clips may be misdetected as a third language (e.g., Urdu transcribed as Chinese).
- **Test method**: Speak Urdu with `language="en"` forced; then with auto-detection; then with the closed-set bilingual approach.
- **Status**: CONFIRMED — observed in testing. `language="en"` produced entirely garbled English for Urdu speech. Auto-detection (`language` param omitted) misdetected Urdu as Chinese on short clips.
  **Resolution history**:
  1. First fix: forced `language="ur"` unconditionally (`UrduGroqSTTService`). Solved the misdetection but made the bot Urdu-only — English input was force-transcribed as Urdu.
  2. Current fix: `BilingualGroqSTTService` transcribes each utterance twice concurrently (`language="en"` and `language="ur"`), and picks whichever result has the higher average segment confidence (`avg_logprob`). Whisper is never given the option to guess a third language, so the original misdetection failure mode can't recur, while both English and Urdu are now correctly supported. Trade-off: 2x Groq STT API calls per turn (run concurrently, so latency impact is minimal — bounded by the slower of the two calls, not the sum).
- **Log excerpt**:
  ```
  CONVO | User: اے آئے کیا چیز ہے    ← correct, chosen via bilingual confidence comparison
  DEBUG | STT bilingual pick: lang=ur conf=-0.312 (en=-1.847 ur=-0.312) text='اے آئے کیا چیز ہے'
  ```

### B2. Code-switching (mixing Urdu and English mid-sentence)
- **Description**: Pakistani Urdu speakers commonly mix English words into Urdu sentences (e.g., "mujhe computer ke baare mein batao"). The bilingual dual-call approach forces the *entire* utterance into either English or Urdu — whichever wins overall confidence — so a sentence that's mostly Urdu with a few English words will still be transcribed entirely in the winning language, and the embedded English/Urdu words may come out garbled.
- **Test method**: Speak a mixed Urdu-English sentence; observe how Whisper handles the minority-language portions.
- **Status**: EXPECTED — the current bilingual approach picks one language for the whole segment, it does not do word-level language switching within a single utterance. Accuracy will degrade for the minority-language portion of a code-switched sentence.

### B3. Dialect variation
- **Description**: Whisper's accuracy varies significantly across regional dialects. Punjabi-accented Urdu, Karachi vs. Lahore Urdu, and formal broadcast Urdu have noticeably different recognition rates.
- **Test method**: Have different speakers test the pipeline; note per-speaker transcription quality.
- **Status**: EXPECTED — not systematically tested.

### B4. Proper nouns and names
- **Description**: Personal names, place names, and brand names are frequently wrong in Whisper output — especially names from South Asian languages rendered in Roman script (e.g., "Zainab" → "Zaineb", "Lahore" → "La Hor").
- **Test method**: Ask the bot about a named entity; check if the name in the transcript matches what was spoken.
- **Status**: EXPECTED — well-known Whisper limitation. No mitigation currently in place.

### B5. Technical jargon and acronyms
- **Description**: Domain-specific terms (medical, legal, engineering) and acronyms are transcription errors waiting to happen. "API" may become "a pie", "LUMS" may become "looms", "ML" may become "M.L." with inconsistent spacing.
- **Test method**: Speak several technical terms and acronyms; check transcript accuracy.
- **Status**: EXPECTED — Whisper's `prompt` parameter can be used to prime it with expected vocabulary. Not currently used.

### B6. Whisper hallucination on short/silent audio
- **Description**: Whisper is known to hallucinate plausible-sounding transcriptions on very short, very quiet, or effectively silent audio segments. Common hallucinations include "Thank you for watching", "Thanks for watching!", or random phrases in whatever language was last detected.
- **Test method**: Trigger a VAD turn by making a very brief non-speech sound; observe transcript.
- **Status**: EXPECTED — documented in OpenAI's Whisper research and well-reported by community. No mitigation currently in place.

---

## C. LLM / Conversation

### C1. LLM outputting unspoken formatting
- **Description**: Llama 3.3 70B may produce markdown formatting (bullet points, bold, headers, code blocks) or emojis in responses despite the system prompt instruction to avoid them. These render as meaningless tokens when spoken aloud by TTS ("asterisk asterisk important asterisk asterisk").
- **Test method**: Ask a question that typically invites a list response (e.g., "give me five tips for..."); check if TTS output includes "asterisk" or "dash" artifacts.
- **Status**: EXPECTED — system prompt includes "avoid emojis, bullet points, or other formatting that can't be spoken" but LLMs do not perfectly follow this instruction 100% of the time.

### C2. LLM outputting Urdu text (Kokoro crash)
- **Description**: When user speaks Urdu and the system prompt does not explicitly enforce English responses, Llama may respond in Urdu Arabic script. Kokoro TTS crashes with `IndexError: index 510 is out of bounds` when attempting to phonemize non-Latin text.
- **Test method**: Remove the "Always respond in English" system prompt instruction; speak Urdu; observe whether Kokoro crashes.
- **Status**: CONFIRMED — observed in testing. Stack trace:
  ```
  WARNING  Phonemes are too long, truncating to 510 phonemes
  IndexError: index 510 is out of bounds for axis 0 with size 510
  ```
  **Resolution**: Added "You understand all languages. Always respond in English regardless of what language the user speaks in." to `SYSTEM_INSTRUCTION`. Urdu text no longer reaches Kokoro.
- **Remaining gap**: Kokoro cannot speak Urdu. If Urdu TTS output is ever needed, swap to Google Cloud TTS (`ur-IN`) or Azure TTS (`ur-PK`).

### C3. Context window overflow in long conversations
- **Description**: Llama 3.3 70B's context window is finite. In a very long conversation, early turns will be truncated. The bot may appear to "forget" things said at the start of the session.
- **Test method**: Hold a 30+ turn conversation; in the final turn, ask about something said in the very first turn.
- **Status**: EXPECTED — no context management or summarization implemented. LLM context grows unbounded until Groq's API rejects the request or silently truncates.

### C4. Ambiguous follow-up / pronoun resolution
- **Description**: User asks a follow-up that relies on shared context: "what about that other one?" or "can you explain it differently?". LLM must correctly resolve the reference from conversation history.
- **Test method**: Have a multi-turn conversation; use vague pronouns in follow-ups.
- **Status**: EXPECTED — depends on LLM capability, not pipeline architecture.

### C5. Empty or near-empty transcript sent to LLM
- **Description**: If VAD fires on a non-speech sound and Whisper produces an empty string or a single filler word ("um", "uh"), the `user_aggregator` still adds it to context and fires the LLM. The LLM responds to an effectively empty turn, producing a confusing "I didn't catch that" or random response.
- **Test method**: Make a very brief non-speech sound near the mic; observe whether the LLM responds.
- **Status**: EXPECTED — `push_empty_transcripts=False` is the default in `BaseWhisperSTTService`, so empty strings are filtered. Single filler words are not filtered and will still reach the LLM.

### C6. Endlessly long LLM response
- **Description**: Open-ended prompts ("explain quantum mechanics in full detail") may produce very long LLM outputs. TTS latency compounds with response length; the user may wait 30+ seconds for audio to complete.
- **Test method**: Ask an open-ended question that invites a long answer.
- **Status**: EXPECTED — no response length limiting implemented. The brevity instruction in `SYSTEM_INSTRUCTION` is a soft constraint.

---

## D. TTS / Output

### D1. Urdu/Arabic script crashing Kokoro
- **Description**: See C2. Kokoro phonemizer overflows on Arabic script characters.
- **Status**: CONFIRMED + RESOLVED (via system prompt forcing English responses).

### D2. Special characters and markdown artifacts in spoken output
- **Description**: Asterisks, underscores, angle brackets, and other markdown symbols from LLM output are spoken literally by Kokoro ("asterisk", "underscore", "greater than").
- **Test method**: Trigger a markdown-formatted LLM response; listen to TTS output.
- **Status**: EXPECTED — no text sanitization layer between LLM and TTS. Mitigation: strengthen system prompt or add a preprocessing step to strip markdown before TTS.

### D3. Numbers and abbreviations
- **Description**: Large numbers (1,000,000), currency ($50), abbreviations (Dr., etc., e.g.), and dates may be spoken incorrectly by Kokoro (e.g., "one comma zero zero zero comma zero zero zero").
- **Test method**: Ask a question whose answer includes a large number or abbreviation.
- **Status**: EXPECTED — Kokoro does not have number normalization. No mitigation currently in place.

### D4. URLs in LLM responses
- **Description**: If the LLM includes a URL in its response, Kokoro will read it character by character or in a way that is entirely unintelligible ("h t t p s colon slash slash...").
- **Test method**: Ask the bot for a website reference.
- **Status**: EXPECTED — system prompt does not explicitly prohibit URLs. No sanitization in place.

---

## E. API / Network Failures

### E1. TTS failure — Cartesia account/billing rejection (HTTP 402)

> **Note**: Cartesia TTS was the original TTS service. This issue was discovered and fixed before switching to Kokoro. Documented for completeness and because the same failure mode applies to any API-based TTS.

- **Description**: What happens when TTS is entirely unreachable — not rate-limited, but rejecting every connection attempt.
- **Test method**: Unplanned/organic — discovered while attempting the barge-in re-test (A4) in `--local` mode. Cartesia's API rejected every WebSocket connection attempt with `HTTP 402 Payment Required`.
- **Result — CONFIRMED, SEVERE**: The `on_pipeline_error` fallback made the situation worse. Because the fallback apology is spoken via `TTSSpeakFrame` through the same broken `CartesiaTTSService`, every fallback attempt immediately failed and re-triggered `on_pipeline_error` — an unbounded retry loop.
  - Observed: **198 failed reconnection attempts in ~65 seconds** (~3/sec) before manual kill.
  - Risk: API key or IP flagged for abuse if left unattended.
  - Log excerpt:
    ```
    ERROR | CartesiaTTSService#0 exception: Unknown error occurred: server rejected WebSocket connection: HTTP 402
    WARNING | PipelineWorker#0: Something went wrong: ErrorFrame#198(...)
    ERROR | Pipeline error from CartesiaTTSService#0: ...
    ```
- **Status**: RESOLVED
  - **Fix**: `on_pipeline_error` now (1) skips the spoken fallback when `frame.processor is tts` — logs only; (2) enforces a 5-second cooldown (`FALLBACK_COOLDOWN_SECS`) between any fallback attempts.
  - **Applies to Kokoro too**: Kokoro is local so HTTP 402 isn't possible, but the same guard protects against any future TTS failure mode.

### E2. STT failure (bad API key)
- **Description**: If `GROQ_API_KEY` is invalid or revoked, every STT request fails with HTTP 401/403. The `on_pipeline_error` fallback should speak an apology.
- **Test method**: Run `$env:GROQ_API_KEY="bad_key"; python bot.py --local`, speak, listen for fallback message.
- **Status**: EXPECTED — fallback guard is in place, but this specific path has not been live-tested.

### E3. LLM failure (bad model name or key)
- **Description**: If the LLM API key is invalid or the model name is wrong, LLM requests fail. Similar to E2.
- **Test method**: Set `model="nonexistent-model"` in `GroqLLMService.Settings`; speak a turn; listen for fallback.
- **Status**: EXPECTED — not tested.

### E4. Rate limiting (HTTP 429)
- **Description**: Groq's free tier has per-minute and per-day token limits. Under sustained use (or very long responses), the pipeline may hit rate limits. The `on_pipeline_error` fallback should fire.
- **Test method**: Send many rapid turns or very long prompts to exhaust the RPM limit.
- **Status**: EXPECTED — same code path as E2/E3. Groq limits are relatively generous for demo use.

### E5. Network dropout mid-stream
- **Description**: If the network connection to Groq's API drops mid-LLM-stream, the streaming response is cut off. TTS may speak an incomplete sentence. The pipeline may hang waiting for a response that never completes.
- **Test method**: Trigger a long LLM response; disable Wi-Fi mid-stream.
- **Status**: EXPECTED — not tested.

---

## F. System / Infrastructure

### F1. Windows console encoding (cp1252) mangling non-ASCII transcripts
- **Description**: Windows console defaults to cp1252. Arabic/Urdu script characters are outside cp1252's range. When loguru writes a Urdu transcript to stderr, the encoding mismatch causes characters to display as `???` or raises a `UnicodeEncodeError`.
- **Test method**: Run the bot without `PYTHONUTF8=1` or the stderr UTF-8 reconfiguration; speak Urdu; observe CONVO log output.
- **Status**: CONFIRMED + RESOLVED
  - **Fix**: `bot.py` now reconfigures `sys.stderr` to UTF-8 at startup before loguru attaches its sink:
    ```python
    if hasattr(sys.stderr, "buffer") and sys.stderr.encoding.lower() != "utf-8":
        import io
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    ```

### F2. Slow startup due to Windows Defender scanning
- **Description**: Windows Defender scans native Python extension files (.pyd/.dll) in the venv on every process launch. With a large venv (pipecat + its deps), startup time ranged from 20-27 seconds inconsistently.
- **Status**: CONFIRMED + RESOLVED
  - **Fix**: Added `D:\venvs\pipecat-voice` to Windows Defender exclusions. Startup time dropped from ~21s to ~3s consistently.

### F3. Audio device disconnection mid-conversation
- **Description**: If the headset is unplugged while the bot is running in `--local` mode, PyAudio will throw an exception. The pipeline may hang, crash, or enter an error loop depending on how pipecat handles the device error.
- **Test method**: Unplug headset mid-conversation in `--local` mode.
- **Status**: EXPECTED — not tested.

### F4. Port 7860 already in use (WebRTC mode)
- **Description**: If a previous bot instance was not cleanly shut down, port 7860 remains bound. A new `python bot.py` start will fail with a "port already in use" error.
- **Test method**: Start the bot twice without killing the first instance.
- **Status**: EXPECTED (known behavior) — fix is `taskkill //F //PID <pid>` to release the port.

### F5. Missing API key at startup
- **Description**: If `GROQ_API_KEY` is not set in `.env`, `os.environ["GROQ_API_KEY"]` raises a `KeyError` at pipeline construction time — but only once the first user turn fires (lazy construction), not at startup. The error is non-obvious.
- **Test method**: Remove `GROQ_API_KEY` from `.env`; start the bot; speak a turn; observe error.
- **Status**: EXPECTED — not tested. Better behavior would be to validate all required keys at startup and fail fast with a clear message.

---

## G. Uncommon / Interesting Edge Cases

### G1. Whisper hallucination on quiet/empty audio
- **Description**: Whisper is documented to hallucinate plausible-sounding text on very short or near-silent audio. Common hallucinations include stock phrases like "Thank you for watching" or "Please subscribe." These reach the LLM as legitimate user turns.
- **Status**: EXPECTED — Whisper research paper and community reports confirm this. No mitigation in place.

### G2. VAD firing on the bot's own voice (without headphones)
- **Description**: In speakerphone mode, TTS audio coming from the speaker can trigger VAD. The bot's own spoken output is captured by the mic, transcribed, and fed back to the LLM — a self-response loop. The loop continues until the pipeline is killed.
- **Status**: EXPECTED — prevented by using wired headphones. Untested in speakerphone configuration.

### G3. LLM responding to its own greeting
- **Description**: On startup, the pipeline injects "Please introduce yourself to the user." and fires `LLMRunFrame`. If the greeting response somehow re-triggers the LLM (e.g., through a misconfigured event handler), the bot could enter a greeting loop.
- **Status**: EXPECTED (low probability) — architecture makes this unlikely but not impossible.

### G4. User asking the bot to speak a different language
- **Description**: User explicitly asks the bot to "speak Urdu" or "respond in French." The LLM may comply (generating non-English text) which then crashes Kokoro or produces garbled audio.
- **Status**: EXPECTED — the system prompt's "Always respond in English" should prevent this, but LLMs do not always obey. No hard enforcement at the TTS input layer.

### G5. Very long user input
- **Description**: User speaks for an extended period (e.g., dictating a long paragraph). Whisper processes the full audio segment; the transcript may be very long. LLM processing time increases; total response latency (STT + LLM + TTS) may exceed 10+ seconds.
- **Status**: EXPECTED — no input length limits in place.

### G6. Bilingual context tracking
- **Description**: User speaks Urdu, bot responds in English. User then says "uss ke baare mein aur batao" ("tell me more about that" in Urdu), referring to the English response. The LLM must track that the Urdu reference points to English content in the conversation history.
- **Status**: EXPECTED — depends on LLM capability. Llama 3.3 70B handles cross-lingual context reasonably but not perfectly.

---

## H. Resolved Issues (Historical Reference)

| Issue | Root Cause | Fix |
|---|---|---|
| Urdu transcribed as garbled English | `language=Language.EN` hardcoded in `GroqSTTService` | Subclassed as `BilingualGroqSTTService` |
| Urdu misdetected as Chinese | Whisper auto-detect unconstrained across all languages, unreliable for short Urdu clips | `BilingualGroqSTTService`: transcribe concurrently as forced `en` and forced `ur`, keep the higher-confidence (`avg_logprob`) result — closed 2-language set, no third-language misdetection possible |
| English input forced into Urdu transcription | Interim fix (`UrduGroqSTTService`) hardcoded `language="ur"` unconditionally, making English input unusable | Replaced with `BilingualGroqSTTService`'s dual-call confidence comparison — see above |
| Kokoro crash on Urdu LLM response | Kokoro phonemizer overflows on Arabic script | System prompt forces English-only LLM responses |
| Console encoding mangling Urdu logs | Windows cp1252 can't encode Arabic Unicode | Reconfigured `sys.stderr` to UTF-8 at startup |
| TTS error loop (198 API calls in 65s) | Fallback apology re-triggered broken TTS, causing infinite retry | Added `frame.processor is tts` guard + 5s cooldown |
| Slow startup (20-27s) | Windows Defender scanning venv `.pyd`/`.dll` files | Added venv to Defender exclusions |
| WASAPI rejecting 16kHz capture | WASAPI shared mode enforces native device sample rate (48kHz) | Switched to DirectSound host API in `_find_headset_mic_device_index()` |
| Groq quota exhaustion (HTTP 429) | Gemini free tier 1,500 RPD limit hit | Switched to Groq LLM (Llama 3.3 70B) |
