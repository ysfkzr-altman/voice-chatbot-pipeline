# Edge Case Testing — Voice AI Pipeline

## Summary

Ongoing QA pass on `bot.py` (Groq STT -> Groq LLM Llama 3.3 70B -> Cartesia TTS, WebRTC or `--local` PyAudio). Findings are added incrementally as each edge case is tested. See `things-to-implement-notes-polished-kahn.md` (plan file) for the full checklist and methodology.

## C. API / Network Failures

### C14. TTS failure — Cartesia account/billing rejection (HTTP 402)

- **Description**: What happens when Cartesia TTS is entirely unreachable (not rate-limited, but rejecting every connection attempt).
- **Test method**: Unplanned/organic — discovered while attempting the barge-in re-test (see A4) in `--local` mode. Cartesia's API rejected every WebSocket connection attempt with `HTTP 402 Payment Required`, indicating a billing/credits issue on the Cartesia account itself (not a code bug).
- **Result — CONFIRMED, SEVERE**: The existing `on_pipeline_error` fallback (added earlier to speak an apology on non-fatal errors so the user isn't left in silence) made the situation *worse*, not better. Because the fallback apology is itself spoken via `TTSSpeakFrame` through the same broken `CartesiaTTSService`, every fallback attempt immediately failed and re-triggered `on_pipeline_error`, which attempted another fallback, which failed again — an unbounded retry loop with no backoff and no cap.
  - Observed: **198 failed reconnection attempts to Cartesia's API in ~65 seconds** (roughly 3/sec, sustained) before the process was manually killed.
  - Risk: left unattended, this would hammer Cartesia's servers indefinitely — plausible risk of the API key or source IP being flagged for abuse.
  - Log excerpt:
    ```
    ERROR | CartesiaTTSService#0 exception: Unknown error occurred: server rejected WebSocket connection: HTTP 402
    WARNING | PipelineWorker#0: Something went wrong: ErrorFrame#198(...)
    ERROR | Pipeline error from CartesiaTTSService#0: Unknown error occurred: server rejected WebSocket connection: HTTP 402
    ```
- **Status**: RESOLVED (code fix applied) + ACTION NEEDED (billing, external)
  - **Code fix applied**: `on_pipeline_error` in `bot.py` now (1) skips the spoken fallback entirely when the failing processor is `tts` itself — logs only, since there's no way to "speak" your way out of a dead TTS connection — and (2) enforces a 5-second cooldown (`FALLBACK_COOLDOWN_SECS`) between any fallback attempts regardless of source, as a general safety net against other repeat-failure patterns (e.g. STT failing on every turn).
  - **Still needs action, separately, not a code issue**: the Cartesia account itself needs its billing/credits checked — the pipeline cannot function with zero working TTS regardless of how gracefully it fails.
  - **Not yet tested**: whether the barge-in re-test (A4) can proceed once Cartesia billing is resolved — this blocked that test entirely this session.
