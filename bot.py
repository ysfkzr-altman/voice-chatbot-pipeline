"""
Rudimentary voice AI chatbot pipeline. Runs either over WebRTC in the browser
or directly against the local mic/speakers from the CLI.

Pipeline: mic (WebRTC or local) -> Silero VAD -> Groq STT (Whisper)
          -> Groq LLM (Llama 3.3 70B) -> Kokoro TTS (local, no API key) -> speakers

Run:
    python bot.py             # WebRTC server; open http://localhost:7860
    python bot.py --local     # Talk directly via the local mic/speakers

Requires GROQ_API_KEY in a .env file.
Kokoro model files (~87 MB) are downloaded automatically on first run to
~/.cache/pipecat/kokoro-onnx/.
"""

import asyncio
import dataclasses
import io
import os
import re
import sys
import time
import wave

import pyaudio
from dotenv import load_dotenv
from loguru import logger
from num2words import num2words

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    AggregatedTextFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSTextFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import EvalRunnerArguments, RunnerArguments, SmallWebRTCRunnerArguments
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.runner.utils import create_transport
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.groq.stt import GroqSTTService
from pipecat.services.whisper.base_stt import Transcription
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.utils.text.base_text_aggregator import AggregationType
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

if "GROQ_API_KEY" not in os.environ:
    sys.exit("GROQ_API_KEY is missing - add it to your .env file.")

# Windows consoles default to cp1252, which can't encode Urdu/Arabic script.
# Reconfigure stderr to UTF-8 before loguru attaches so transcripts print
# correctly instead of crashing or showing ???.
if hasattr(sys.stderr, "buffer") and sys.stderr.encoding.lower() != "utf-8":
    import io
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logger.remove(0)

# A dedicated level for just the conversation transcript (what the user said,
# what the bot said) - separate from DEBUG/INFO noise. Must be registered
# before any sink references it by name.
logger.level("CONVO", no=25, color="<green>", icon="")
# Set LOG_LEVEL=DEBUG in the environment to also see per-turn diagnostics
# (e.g. the raw STT transcript logged in UrduGroqSTTService._transcribe).
logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "DEBUG"))

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant in a voice conversation. Your responses will "
    "be spoken aloud, so avoid emojis, bullet points, or other formatting that "
    "can't be spoken. "
    "You understand all languages. Always respond in English regardless of what language the user speaks in. "
    "Keep every response short and direct — one or two "
    "sentences by default, never more than a few. No filler, no preamble, no "
    "restating the question, no hedging caveats. Answer exactly what was asked "
    "and stop. "
    "If the user's message is too short, vague, or ambiguous to answer "
    "meaningfully (e.g. a single word like 'yes' or 'ok' with no clear "
    "context), don't guess at what they might mean - ask a brief clarifying "
    "question instead."
)

GREETING_MESSAGE = "Hi! How can I help you today?"
FALLBACK_ERROR_MESSAGE = "Sorry, I hit a glitch there. Could you say that again?"
FALLBACK_COOLDOWN_SECS = 5.0

KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_heart")

# Bulletproofing: if Kokoro TTS ITSELF is what's failing (corrupted model,
# crash, resource exhaustion), speaking a fallback apology through that
# same broken TTS just triggers another failure - previously handled by
# giving up silently (the user hears nothing at all, with no indication
# anything went wrong). This is a pre-recorded WAV, generated once
# offline via Kokoro itself and committed to the repo, played by pushing
# its raw PCM samples directly to the transport - it never calls Kokoro's
# synthesis pipeline, so it works even when that pipeline is what's
# broken. See scripts/generate_fallback_audio.py to regenerate it.
FALLBACK_AUDIO_PATH = os.path.join(os.path.dirname(__file__), "fallback_audio.wav")


def _load_fallback_audio() -> tuple[bytes, int, int] | None:
    """Loads the pre-recorded fallback WAV's raw PCM bytes + format.

    Returns None (rather than raising) if the file is missing/unreadable -
    a missing fallback file shouldn't itself crash bot startup, though it
    does mean this specific safety net is unavailable.
    """
    try:
        with wave.open(FALLBACK_AUDIO_PATH, "rb") as wf:
            audio = wf.readframes(wf.getnframes())
            return audio, wf.getframerate(), wf.getnchannels()
    except OSError as e:
        logger.error(f"Could not load fallback audio ({FALLBACK_AUDIO_PATH}): {e}")
        return None


FALLBACK_AUDIO = _load_fallback_audio()

# Confirmed problem #12: very short utterances (backchannel-style sounds
# like "Mm-hmm", or short interruption phrases like "Wait, stop") kept
# getting misclassified as Urdu gibberish instead of the English that was
# actually said - short/ambiguous audio gives Whisper's Urdu pass just
# enough room to produce a confidently-wrong result. Applied as a bonus to
# the English candidate's confidence score, only when the winning
# transcript is short enough to plausibly be one of these cases.
SHORT_UTTERANCE_CHAR_THRESHOLD = 12
# 0.3 wasn't enough in practice - measured the real gap for "Mm-hmm"
# directly (en=-0.717 vs ur=-0.212, a 0.5 gap) and it still lost. 0.7
# comfortably covers the measured case with some margin.
SHORT_UTTERANCE_EN_BIAS = 0.7


def _avg_logprob(result: Transcription) -> float:
    """Mean segment log-probability, used as a confidence proxy.

    Whisper doesn't expose a single "confidence" number, but each segment's
    avg_logprob (closer to 0 = more confident) is the standard stand-in.
    Returns -inf for silent/empty audio (no segments) so it always loses to
    a real transcript when comparing two candidates.
    """
    segments = getattr(result, "segments", None) or []
    if not segments:
        return float("-inf")
    return sum(getattr(s, "avg_logprob", 0.0) for s in segments) / len(segments)


class BilingualGroqSTTService(GroqSTTService):
    """GroqSTTService constrained to a closed set of two languages: English and Urdu.

    Whisper's own language auto-detection was tried and misdetected short
    Urdu clips as Chinese - it's unconstrained across every language Whisper
    knows, which is overkill and unreliable for a bot that only ever needs
    to distinguish two specific languages.

    Instead, each utterance is transcribed twice concurrently - once forced
    to `language="en"`, once forced to `language="ur"` - and whichever
    result has the higher average segment confidence (avg_logprob) wins.
    Forcing removes the third-language misdetection failure mode entirely,
    since Whisper is never given the option to guess anything else.

    Trade-off: this doubles Groq STT API calls per turn. Both calls run
    concurrently via asyncio.gather, so wall-clock latency is roughly the
    slower of the two, not the sum - but token/request usage is 2x.

    NOTE: The LLM is instructed (system prompt) to always reply in English
    regardless of input language, because the TTS side (KokoroTTSService)
    does not support Urdu script and will error on it. This means the bot
    understands both languages but only speaks English. To make it speak
    Urdu too, swap KokoroTTSService for Google Cloud TTS (ur-IN) or Azure
    TTS (ur-PK), both of which have Urdu voices.
    """

    async def _transcribe(self, audio: bytes) -> Transcription:
        base_kwargs = {
            "file": ("audio.wav", audio, "audio/wav"),
            "model": self._settings.model,
            "response_format": "verbose_json",
        }
        if self._settings.prompt is not None:
            base_kwargs["prompt"] = self._settings.prompt
        if self._settings.temperature is not None:
            base_kwargs["temperature"] = self._settings.temperature

        result_en, result_ur = await asyncio.gather(
            self._client.audio.transcriptions.create(language="en", **base_kwargs),
            self._client.audio.transcriptions.create(language="ur", **base_kwargs),
        )

        conf_en, conf_ur = _avg_logprob(result_en), _avg_logprob(result_ur)

        # Bias toward English for short candidates - see
        # SHORT_UTTERANCE_EN_BIAS above for why. Based on the English
        # candidate's own length, not Urdu's, so a genuinely short Urdu
        # utterance doesn't get penalized by comparison to a long
        # (likely-wrong) English guess for the same audio.
        biased_conf_en = conf_en
        if len(result_en.text.strip()) <= SHORT_UTTERANCE_CHAR_THRESHOLD:
            biased_conf_en += SHORT_UTTERANCE_EN_BIAS

        winner, lang, conf = (
            (result_en, "en", conf_en)
            if biased_conf_en >= conf_ur
            else (result_ur, "ur", conf_ur)
        )

        # repr() is ASCII-safe: shows \uXXXX escapes for non-Latin chars so
        # you can tell whether Whisper returned actual Urdu Unicode or English.
        logger.debug(
            f"STT bilingual pick: lang={lang} conf={conf:.3f} "
            f"(en={conf_en:.3f} ur={conf_ur:.3f}) text={repr(winner.text)}"
        )
        return winner


# Confirmed problem #10: nothing enforced that LLM output stays
# speech-friendly beyond asking nicely in the system prompt, which isn't
# reliable - confirmed with a real example where the LLM produced the
# literal text "1,2,3,4,5" instead of speaking the numbers as words.
# Matches either a short comma-separated list of small numbers (read as
# separate numbers, e.g. "1,2,3,4,5" -> "one, two, three, four, five") or
# any other standalone number (read as one number, so "8,499,000" becomes
# "eight million, four hundred and ninety-nine thousand" instead of a
# raw digit string with commas in it, which doesn't read naturally aloud).
_NUMBER_LIST_PATTERN = re.compile(r"\b\d{1,2}(?:,\d{1,2}){2,}\b")
_NUMBER_PATTERN = re.compile(r"\b\d[\d,]*\b")

# Found while bulletproofing: asking for "a website, just give me the URL"
# made the LLM produce a raw URL (e.g.
# "https://docs.python.org/three/tutorial/index.html") straight into TTS.
# Beyond being unintelligible read aloud character-by-character, the
# unusual punctuation density measurably slowed Kokoro's synthesis (5.3s
# TTFB for that one line, vs. a typical ~1-2s) - a real latency/reliability
# risk, not just a quality one. Replaced wholesale with a short spoken
# placeholder rather than trying to "read out" the domain - a URL's path/
# query string can contain arbitrary other unusual characters a
# domain-only conversion wouldn't protect against.
_URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)

# Polish pass found while auditing _normalize_for_speech: the generic
# _NUMBER_PATTERN above treats digits as one big cardinal number regardless
# of context, which is wrong for several common shapes an LLM answer can
# contain. Each of these runs BEFORE _NUMBER_PATTERN so its digits are
# already consumed (turned into words) by the time the generic pattern
# would otherwise mangle them.
#   - "$50"           -> "one big number" bug: dangling "$" was left in
#                        place and spoken literally ("dollar sign fifty").
#   - "95%"            -> same dangling-symbol problem with "%".
#   - "3.14"           -> generic pattern only matches digits, so "3" and
#                        "14" got converted separately either side of a
#                        literal spoken period: "three.fourteen".
#   - "555-123-4567"   -> read as one gigantic cardinal number instead of
#                        digit-by-digit, which is how phone numbers are
#                        actually spoken.
#   - "3:30"           -> same problem as decimals, via ":" instead of ".".
# Verified each independently against the previous behavior (see the
# bash repro used while diagnosing this) before adding.
_CURRENCY_PATTERN = re.compile(r"\$\d[\d,]*(?:\.\d{1,2})?")
_PERCENT_PATTERN = re.compile(r"\b\d[\d,]*(?:\.\d+)?%")
_PHONE_PATTERN = re.compile(r"\b\d{3}-\d{3}-\d{4}\b")
_TIME_PATTERN = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b(\s*[ap]\.?m\.?)?", re.IGNORECASE)
_DECIMAL_PATTERN = re.compile(r"\b\d+\.\d+\b")


def _digits_to_words(digits: str) -> str:
    return " ".join(num2words(int(d)) for d in digits)


def _replace_currency(match: re.Match) -> str:
    amount = match.group(0)[1:].replace(",", "")
    dollars_str, _, cents_str = amount.partition(".")
    dollars = int(dollars_str) if dollars_str else 0
    words = f"{num2words(dollars)} dollars"
    if cents_str:
        cents = int(cents_str.ljust(2, "0")[:2])
        if cents:
            words += f" and {num2words(cents)} cents"
    return words


def _replace_percent(match: re.Match) -> str:
    num_str = match.group(0)[:-1].replace(",", "")
    whole_str, _, frac_str = num_str.partition(".")
    words = num2words(int(whole_str)) if whole_str else "zero"
    if frac_str:
        words += f" point {_digits_to_words(frac_str)}"
    return f"{words} percent"


def _replace_phone(match: re.Match) -> str:
    return _digits_to_words(match.group(0).replace("-", ""))


def _replace_time(match: re.Match) -> str:
    hour, minute, suffix = int(match.group(1)), int(match.group(2)), match.group(3)
    hour_words = num2words(hour if hour else 12)
    if minute == 0:
        minute_words = "o'clock"
    elif minute < 10:
        minute_words = f"oh {num2words(minute)}"
    else:
        minute_words = num2words(minute)
    result = f"{hour_words} {minute_words}"
    if suffix:
        result += " " + re.sub(r"\.", "", suffix.strip()).lower()
    return result


def _replace_decimal(match: re.Match) -> str:
    whole, _, frac = match.group(0).partition(".")
    return f"{num2words(int(whole))} point {_digits_to_words(frac)}"


def _normalize_for_speech(text: str) -> str:
    def _replace_list(match: re.Match) -> str:
        return ", ".join(num2words(int(part)) for part in match.group(0).split(","))

    def _replace_number(match: re.Match) -> str:
        try:
            return num2words(int(match.group(0).replace(",", "")))
        except ValueError:
            return match.group(0)

    text = _URL_PATTERN.sub("a link", text)
    text = _CURRENCY_PATTERN.sub(_replace_currency, text)
    text = _PERCENT_PATTERN.sub(_replace_percent, text)
    text = _PHONE_PATTERN.sub(_replace_phone, text)
    text = _TIME_PATTERN.sub(_replace_time, text)
    text = _DECIMAL_PATTERN.sub(_replace_decimal, text)
    text = _NUMBER_LIST_PATTERN.sub(_replace_list, text)
    text = _NUMBER_PATTERN.sub(_replace_number, text)
    return text


# Confirmed problem #3 (partial fix - see the reverted commit above for
# the part that couldn't be fixed): pipecat's interruption mechanism
# stops the bot's audio the moment VAD detects speech, before STT even
# runs, so nothing at this layer can prevent the acoustic cutoff itself.
# What this DOES fix: once cut off, the backchannel sound used to get
# sent to the LLM and treated as a real question. Deliberately restricted
# to genuinely non-lexical filler sounds - excludes real words like
# "yeah"/"okay"/"right" that could be a legitimate short answer, since
# silently dropping those would conflict with the ambiguous-input fix
# above (which asks for clarification rather than ignoring a short
# reply).
# Matched after stripping everything but letters (see BackchannelFilter),
# so e.g. Whisper's "M.M. Humm." normalizes to "mmhumm" before comparing -
# these entries are deliberately written in that same stripped form.
_BACKCHANNEL_WORDS = {
    "mm", "mmhmm", "mmhumm", "hmm", "hm", "uhhuh", "mhm",
}

# EDGE_CASES.md B6/G1: Whisper is documented to hallucinate plausible-
# sounding stock phrases on short, quiet, or near-silent audio (a VAD
# false-trigger on a cough, a chair creak, etc.). Unlike backchannel
# filler, these come out as complete, grammatical sentences - not empty
# and not a recognizable filler word - so neither push_empty_transcripts
# nor _BACKCHANNEL_WORDS catches them, and the bot answers a turn nobody
# actually said. A confidence-score cutoff was tried and rejected earlier
# in this project (measured directly: hallucinated text on silence scores
# competitively with real speech from this Whisper+Groq setup, so
# avg_logprob can't distinguish them). This instead denylists the specific
# stock phrases that are well-documented as Whisper's go-to hallucinations
# (YouTube-outro-style boilerplate) - deliberately narrow and exact-match,
# so it can't accidentally swallow a real, on-topic reply.
_HALLUCINATION_PHRASES = {
    "thank you for watching",
    "thanks for watching",
    "thank you for watching this video",
    "please subscribe",
    "subscribe to my channel",
    "dont forget to subscribe",
    "like and subscribe",
    "see you in the next video",
    "ill see you in the next video",
    "thanks for listening",
}


class BackchannelFilter(FrameProcessor):
    """Drops transcribed backchannel filler sounds and known Whisper
    hallucination phrases before they reach the user-turn aggregator, so
    neither is ever sent to the LLM as if it were a real question. Placed
    between `stt` and `user_aggregator`.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            lowered = frame.text.lower()

            # Strip everything but letters - Whisper's punctuation choices
            # for filler sounds are inconsistent ("Mm-hmm" vs "M.M. Humm."
            # vs "Mmhmm.") so match on letters alone.
            letters_only = re.sub(r"[^a-z]", "", lowered)
            if letters_only in _BACKCHANNEL_WORDS:
                logger.debug(f"Dropping backchannel-only transcript: {frame.text!r}")
                return

            # Same idea but keeping word boundaries, since these are
            # multi-word phrases rather than single filler sounds. Substring
            # containment, not exact equality: real hallucinated transcripts
            # vary in surrounding wording ("Thanks for watching this video!"
            # vs "Thanks for watching." vs "...watching, please subscribe!")
            # while still reliably containing one of these distinctive
            # markers verbatim - confirmed exact-match missed real variants
            # ("Please subscribe to my channel." didn't equal either
            # "please subscribe" or "subscribe to my channel" on its own).
            words_only = " ".join(re.sub(r"[^a-z\s]", "", lowered).split())
            if any(phrase in words_only for phrase in _HALLUCINATION_PHRASES):
                logger.debug(f"Dropping known Whisper hallucination phrase: {frame.text!r}")
                return
        await self.push_frame(frame, direction)


# Confirmed problem #6: Kokoro has a hard ~510-phoneme limit per request
# and crashes (IndexError) instead of truncating gracefully when it's
# exceeded. The max_completion_tokens cap on the LLM (see GroqLLMService
# below) makes a single massive response unlikely, but doesn't rule out
# one very long individual sentence with no punctuation to break on -
# Kokoro's own internal sentence aggregation only splits on sentence
# boundaries, not length. This is a hard backstop measured in characters
# (not phonemes, which aren't cheap to compute here), picked
# conservatively low relative to the ~510-phoneme limit so it trips well
# before that regardless of how phoneme-dense the text is.
KOKORO_MAX_CHUNK_CHARS = 300


def _split_for_tts_safety(text: str) -> list[str]:
    """Splits `text` into chunks no longer than KOKORO_MAX_CHUNK_CHARS,
    breaking on word boundaries so words are never cut mid-way."""
    if len(text) <= KOKORO_MAX_CHUNK_CHARS:
        return [text]

    chunks = []
    current = ""
    for word in text.split(" "):
        candidate = f"{current} {word}".strip()
        if len(candidate) > KOKORO_MAX_CHUNK_CHARS and current:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


class TTSTextNormalizer(FrameProcessor):
    """Rewrites text into speech-friendly form right before TTS, and splits
    any chunk that's too long for Kokoro into smaller pieces.

    A backup for the system prompt's "keep it speakable" instruction, which
    isn't reliable on its own (see the module comment above `_NUMBER_LIST_PATTERN`).
    Placed between `llm` and `tts` in the pipeline so it sees exactly what's
    about to be spoken, regardless of whether it came from the LLM or a
    direct `TTSSpeakFrame` (greeting/fallback).

    The LLM streams its response as many small text fragments (sometimes a
    single word or less per frame), not one complete string - confirmed
    directly (bot.py's own CONVO log) that checking/transforming each
    fragment independently doesn't work: a number-list spanning multiple
    fragments never matches the list pattern, and a long response spread
    across many short fragments never trips the length-based chunk split,
    since Kokoro's own internal aggregator (`TTSService`/`SimpleTextAggregator`)
    re-assembles the fragments into complete sentences *after* this
    processor runs. Mirrors that same buffering here with another
    `SimpleTextAggregator`, so normalization and length-splitting run on
    complete sentences - what Kokoro will actually receive - not
    fragments. Re-emits already-aggregated `AggregatedTextFrame`s so
    `TTSService` uses them as-is instead of re-aggregating.
    """

    def __init__(self):
        super().__init__()
        self._aggregator = SimpleTextAggregator()

    async def _emit_normalized(self, text: str, direction: FrameDirection):
        normalized = _normalize_for_speech(text)
        chunks = _split_for_tts_safety(normalized)
        if len(chunks) > 1:
            logger.debug(
                f"Splitting {len(normalized)}-char sentence into {len(chunks)} "
                f"pieces before TTS (over KOKORO_MAX_CHUNK_CHARS)."
            )
        for chunk in chunks:
            await self.push_frame(
                AggregatedTextFrame(chunk, AggregationType.SENTENCE, raw_text=chunk), direction
            )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            # Discard any partially-buffered text from a response that got
            # cut off - otherwise leftover text could leak into and corrupt
            # the start of the next turn's response.
            await self._aggregator.handle_interruption()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TTSSpeakFrame) and frame.text:
            # Already a complete, one-shot utterance (greeting/fallback) -
            # no streaming fragments to buffer.
            normalized = _normalize_for_speech(frame.text)
            for chunk in _split_for_tts_safety(normalized):
                await self.push_frame(dataclasses.replace(frame, text=chunk), direction)
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            remaining = await self._aggregator.flush()
            if remaining and remaining.text:
                await self._emit_normalized(remaining.text, direction)
            await self.push_frame(frame, direction)
            return

        if (
            isinstance(frame, TextFrame)
            and not isinstance(frame, AggregatedTextFrame)
            and frame.text
        ):
            async for aggregation in self._aggregator.aggregate(frame.text):
                await self._emit_normalized(aggregation.text, direction)
            return

        await self.push_frame(frame, direction)


class ConservativeSmartTurnAnalyzer(LocalSmartTurnAnalyzerV3):
    """LocalSmartTurnAnalyzerV3 with a stricter completion threshold.

    Confirmed problems #8 (a second turn-taking model - "Smart Turn" -
    silently loads by default and can override how long the bot waits for
    silence) and #11 (a single sentence with a natural mid-sentence pause
    got cut into two separate turns, triggering a reply to only the first
    half). Both trace to the same root cause: pipecat's own
    `_predict_endpoint` hardcodes `prediction = 1 if probability > 0.5
    else 0` with no constructor param to adjust it - a bare coin-flip
    threshold for "is this sentence actually finished". Reproduced exactly
    this failure: it predicted "complete" for audio ending mid-sentence
    (a trailing comma before a continuing clause).

    Requiring a higher-confidence probability before accepting "complete"
    makes the model more conservative - biased toward waiting a bit longer
    rather than cutting the user off mid-thought. Trade-off: in genuinely
    ambiguous cases, the bot may pause slightly longer before responding
    than it would with the stock 0.5 threshold.

    Threshold picked from direct measurement, not guessed: fed real
    Kokoro-synthesized audio straight to `_predict_endpoint` for both a
    genuinely complete short utterance ("Yes." -> 0.8548) and the
    confirmed incomplete mid-sentence fragment ("Write the numbers 1
    through 5 using digits," -> 0.7434). An initial guess of 0.85 sat
    right on top of "Yes."'s score - in the live pipeline this pushed
    "Yes." just under the threshold, triggering Smart Turn's own internal
    3-second silence fallback instead of a near-instant response. 0.80
    keeps the fragment rejected with real margin on both sides.
    """

    COMPLETION_THRESHOLD = 0.80

    def _predict_endpoint(self, audio_array):
        result = super()._predict_endpoint(audio_array)
        result["prediction"] = 1 if result["probability"] > self.COMPLETION_THRESHOLD else 0
        return result


async def _startup_self_check(
    stt: "BilingualGroqSTTService", llm: GroqLLMService, tts: KokoroTTSService
) -> None:
    """Validates STT, LLM, and TTS actually work before accepting real
    traffic - not just that a key is present (already checked at import
    time), but that a real call to each service succeeds.

    A revoked/expired key, a Groq quota/permission issue scoped to just
    one endpoint, or a corrupted local Kokoro model would otherwise only
    surface on the user's FIRST real turn, presenting as a mysterious
    fallback apology with no clear cause. Failing fast here with a
    specific error per service is much easier to diagnose than that.

    Exits the process (does not raise) if any check fails, matching the
    existing fail-fast behavior for a missing API key / port conflict.
    """
    errors = []

    try:
        await llm._client.chat.completions.create(
            model=llm._settings.model,
            messages=[{"role": "user", "content": "hi"}],
            max_completion_tokens=1,
        )
    except Exception as e:
        errors.append(f"LLM (Groq chat completions, model={llm._settings.model}): {e}")

    if FALLBACK_AUDIO is not None:
        try:
            audio, sample_rate, num_channels = FALLBACK_AUDIO
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(num_channels)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio)
            await stt._client.audio.transcriptions.create(
                file=("check.wav", buf.getvalue(), "audio/wav"),
                model=stt._settings.model,
                language="en",
            )
        except Exception as e:
            errors.append(f"STT (Groq Whisper, model={stt._settings.model}): {e}")
    else:
        errors.append("STT check skipped - no fallback_audio.wav available to test with")

    try:
        async for _ in tts.run_tts("Hi", context_id="startup-self-check"):
            pass
    except Exception as e:
        errors.append(f"TTS (Kokoro, voice={tts._settings.voice}): {e}")

    if errors:
        logger.error("Startup self-check FAILED - fix these before running the bot:")
        for err in errors:
            logger.error(f"  - {err}")
        sys.exit(1)

    logger.info("Startup self-check passed: STT, LLM, and TTS are all working.")


# Bug found while auditing this file for further hardening: pipecat spawns a
# fresh run_bot() per WebRTC connection (background_tasks.add_task - see the
# idle_timeout_secs comment below for the source), so without this flag every
# single connection re-ran the self-check - 3 extra API round-trips (an LLM
# call, an STT call, and a full Kokoro synthesis) added to that user's first-
# turn latency, plus 3x the intended quota usage per session instead of once
# per process. The services themselves (same API key, same models) don't
# change between connections, so verifying them once per process is enough.
_startup_self_check_done = False


async def run_bot(transport: BaseTransport, *, handle_sigint: bool = False):
    # Retry-with-backoff on transient STT/LLM failures (a dropped
    # connection, a momentary 5xx) already happens here, not something
    # this file adds: both use an AsyncOpenAI client under the hood (Groq
    # is OpenAI-API-compatible), which defaults to max_retries=2 with its
    # own exponential backoff. Verified this is real, not just present in
    # the signature: a connection failure with max_retries=0 failed in
    # 2.6s; the default took 8.0s, consistent with multiple attempts
    # actually happening. Not overridden higher - pipecat's create_client
    # methods for both services accept **kwargs but don't forward them to
    # the AsyncOpenAI(...) constructor in this version, so a client-level
    # override would require subclassing purely to keep the same value
    # this default already provides.
    stt = BilingualGroqSTTService(
        api_key=os.environ["GROQ_API_KEY"],
        settings=GroqSTTService.Settings(model="whisper-large-v3"),
    )

    tts = KokoroTTSService(
        settings=KokoroTTSService.Settings(voice=KOKORO_VOICE),
    )

    tts_text_normalizer = TTSTextNormalizer()
    backchannel_filter = BackchannelFilter()

    llm = GroqLLMService(
        api_key=os.environ["GROQ_API_KEY"],
        settings=GroqLLMService.Settings(
            model="llama-3.3-70b-versatile",
            system_instruction=SYSTEM_INSTRUCTION,
            # Hard cap backing up the "keep it short" system prompt instruction -
            # bounds worst-case LLM generation time and TTS synthesis time, and
            # keeps sentences well under Kokoro's ~510-phoneme truncation limit.
            max_completion_tokens=150,
        ),
    )

    # Uses the same, fully-configured stt/llm/tts instances the pipeline
    # will actually run with - not throwaway copies - so the check
    # reflects the exact models/settings in use.
    #
    # Escape hatch for evals/stt_failure_test.yaml specifically: that test
    # deliberately starts the bot with a broken GROQ_API_KEY to verify the
    # mid-turn fallback-apology path still works when STT/LLM fail - this
    # check would otherwise catch that same bad key here and exit before
    # the scenario it's testing ever gets a chance to run.
    global _startup_self_check_done
    if os.getenv("SKIP_STARTUP_SELF_CHECK"):
        logger.warning("SKIPPING startup self-check (SKIP_STARTUP_SELF_CHECK is set).")
    elif _startup_self_check_done:
        logger.debug(
            "Skipping startup self-check - already verified once this process."
        )
    else:
        await _startup_self_check(stt, llm, tts)
        _startup_self_check_done = True

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            # Explicit, deliberate choice (was previously an unexamined
            # default - see ConservativeSmartTurnAnalyzer docstring).
            user_turn_strategies=UserTurnStrategies(
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=ConservativeSmartTurnAnalyzer())]
            ),
        ),
        # Bulletproofing: a long-running conversation would otherwise grow
        # LLMContext unboundedly - every future turn resends the entire
        # history, so cost/latency creep up forever and eventually risk
        # hitting the model's real context limit. pipecat already ships a
        # complete summarization mechanism for this (handles tool-call
        # boundaries, preserves the system prompt) - it's just OFF by
        # default (enable_auto_context_summarization=False). Turning it on
        # with its own sensible defaults (compress once >8000 estimated
        # tokens or >20 unsummarized messages, keeping the last 4 messages
        # verbatim) rather than leaving this pipeline vulnerable to
        # unbounded growth.
        assistant_params=LLMAssistantAggregatorParams(enable_auto_context_summarization=True),
    )

    @user_aggregator.event_handler("on_user_turn_message_added")
    async def on_user_turn_message_added(aggregator, message):
        logger.log("CONVO", f"User: {message.content}")

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message):
        logger.log("CONVO", f"Bot: {message.content}")
        nonlocal consecutive_fallback_failures, circuit_open_until
        # A real assistant turn completed successfully - close the circuit
        # breaker below entirely, so a transient blip doesn't leave things
        # backed off longer than necessary once the service has recovered.
        consecutive_fallback_failures = 0
        circuit_open_until = 0.0

        if message.interrupted and message.content:
            # Confirmed problem #9: when a response is cut off, pipecat
            # still hands this handler the FULL generated text - and right
            # after this handler returns, it broadcasts that same full
            # text into the LLM context (LLMContextAssistantTurnFrame),
            # regardless of how much was actually spoken before the
            # interruption. Reproduced directly (a standalone context-dump
            # probe): interrupting well under a second into a multi-sentence
            # answer still left the complete, uncut text in context - the
            # bot's memory claimed to have said things the user never
            # actually heard.
            #
            # There's no clean hook to prevent that broadcast (it happens
            # in pipecat's own code, after this handler, using a local
            # variable this handler can't reach) or to know precisely how
            # much audio was actually played. This corrects the record
            # after the fact instead: find the just-broadcast entry by its
            # (still-unique-enough) exact text and replace it with an
            # explicit marker, so a later turn can't be misled by a
            # fabricated "I already told you that" when it wasn't heard.
            original_content = message.content

            async def _correct_interrupted_context():
                await asyncio.sleep(0.2)
                messages = context.get_messages()
                for i in reversed(range(len(messages))):
                    if (
                        messages[i].get("role") == "assistant"
                        and messages[i].get("content") == original_content
                    ):
                        messages[i]["content"] = (
                            "[This response was interrupted by the user before finishing - "
                            "only part of it was actually heard, not the complete answer.]"
                        )
                        context.set_messages(messages)
                        logger.debug("Corrected interrupted assistant turn in context")
                        break

            asyncio.create_task(_correct_interrupted_context())

    pipeline = Pipeline(
        [
            transport.input(),  # Mic input
            stt,  # Speech -> text
            backchannel_filter,  # Drop backchannel-only transcripts
            user_aggregator,  # Collect user turn
            llm,  # Generate response
            tts_text_normalizer,  # Clean up text before it's spoken
            tts,  # Text -> speech
            transport.output(),  # Speaker output
            assistant_aggregator,  # Collect assistant turn
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        # Bulletproofing: made explicit rather than relying on the
        # library default. If a connected user goes quiet for this long
        # (no BotSpeakingFrame/UserSpeakingFrame at all), pipecat cancels
        # this session's worker AND runner automatically - confirmed via
        # pipecat's own runner source (background_tasks.add_task spawns
        # an independent bot()/run_bot()/WorkerRunner() per WebRTC
        # connection) that this is scoped to just that one connection,
        # not the shared FastAPI server: an idle user's session cleanly
        # tears down without affecting anyone else or requiring a
        # restart. For --local mode there's only ever one worker for the
        # whole process, so the same timeout just exits the CLI after
        # this long with no mic activity - reasonable for a tool nobody's
        # using anymore.
        idle_timeout_secs=300.0,
    )

    # Confirmed problem #7: a flat per-attempt cooldown still lets the
    # fallback fire indefinitely during a genuine outage (every
    # FALLBACK_COOLDOWN_SECS, forever). This is a real circuit breaker
    # instead: each consecutive failure doubles how long the circuit stays
    # open (backoff grows 5s -> 10s -> 20s -> ... capped at
    # CIRCUIT_BREAKER_MAX_BACKOFF_SECS), and a single successful turn
    # (on_assistant_turn_stopped above) closes it again completely.
    consecutive_fallback_failures = 0
    circuit_open_until = 0.0
    CIRCUIT_BREAKER_MAX_BACKOFF_SECS = 60.0

    @worker.event_handler("on_pipeline_error")
    async def on_pipeline_error(worker, frame: ErrorFrame):
        nonlocal consecutive_fallback_failures, circuit_open_until
        logger.error(f"Pipeline error from {frame.processor}: {frame.error}")
        if frame.fatal:
            return

        now = time.monotonic()
        if now < circuit_open_until:
            # Already backed off from a recent run of failures - stay quiet
            # rather than speaking (and failing) again immediately.
            return

        if frame.processor is tts:
            # Speaking a fallback apology through the TTS engine that's
            # itself broken would just trigger another ErrorFrame - an
            # infinite retry loop hammering it. Play the pre-recorded WAV
            # instead: raw PCM samples pushed straight to the transport,
            # never touching Kokoro's synthesis pipeline at all.
            if FALLBACK_AUDIO is None:
                logger.error(
                    "TTS itself is failing and no fallback audio is available "
                    "- giving up silently for this turn."
                )
                return
            # A bare OutputAudioRawFrame isn't recognized by the transport's
            # speaking-detection logic (only TTSAudioRawFrame/
            # SpeechOutputAudioRawFrame are - confirmed directly: the audio
            # never triggered "bot started speaking" without this). The
            # Started/Stopped pair mirrors what a real TTS service emits
            # around its own audio, so the transport's start/stop-speaking
            # bookkeeping behaves the same as for a normal spoken reply.
            audio, sample_rate, num_channels = FALLBACK_AUDIO
            await worker.queue_frames(
                [
                    TTSStartedFrame(),
                    TTSTextFrame(FALLBACK_ERROR_MESSAGE, AggregationType.SENTENCE),
                    TTSAudioRawFrame(audio=audio, sample_rate=sample_rate, num_channels=num_channels),
                    TTSStoppedFrame(),
                ]
            )
        else:
            # A non-fatal ErrorFrame (STT/LLM hiccup, rate limit, etc.) would
            # otherwise just get logged, leaving the user hearing silence for
            # that turn. Speak a short apology instead so the conversation
            # can continue. append_to_context=False keeps it out of the LLM
            # history.
            await worker.queue_frames(
                [TTSSpeakFrame(text=FALLBACK_ERROR_MESSAGE, append_to_context=False)]
            )

        consecutive_fallback_failures += 1
        backoff = min(
            FALLBACK_COOLDOWN_SECS * (2 ** (consecutive_fallback_failures - 1)),
            CIRCUIT_BREAKER_MAX_BACKOFF_SECS,
        )
        circuit_open_until = now + backoff
        logger.error(
            f"Circuit breaker: {consecutive_fallback_failures} consecutive failure(s), "
            f"backing off spoken fallback for {backoff:.0f}s."
        )

    # Speak a fixed greeting directly instead of round-tripping through the LLM
    # just to say hello - saves an API call and the greeting is heard sooner.
    # append_to_context defaults to True, so it's still recorded in history for
    # any follow-up that references it.
    await worker.queue_frames([TTSSpeakFrame(text=GREETING_MESSAGE)])

    runner = WorkerRunner(handle_sigint=handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Entry point used by the WebRTC dev runner (``python bot.py -t webrtc``)
    and the eval harness (``python bot.py -t eval``)."""
    if isinstance(runner_args, SmallWebRTCRunnerArguments):
        # Imported here (not at module level) so `--local` mode never pays the
        # aiortc import cost - it doesn't use this transport at all.
        from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

        transport = SmallWebRTCTransport(
            webrtc_connection=runner_args.webrtc_connection,
            params=TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
            ),
        )
    elif isinstance(runner_args, EvalRunnerArguments):
        from pipecat.evals.transport import EvalTransportParams

        transport = await create_transport(
            runner_args,
            {"eval": lambda: EvalTransportParams(audio_in_enabled=True, audio_out_enabled=True)},
        )
    else:
        raise RuntimeError(
            "This bot only supports the WebRTC transport (-t webrtc) or eval transport (-t eval)."
        )

    await run_bot(transport, handle_sigint=runner_args.handle_sigint)


def _find_headset_mic_device_index() -> int | None:
    """Look up the headset mic's PyAudio device index on the DirectSound host API.

    DirectSound handles simultaneous input+output more reliably on Windows
    than the MME backend PyAudio defaults to, and - unlike WASAPI's shared
    mode - accepts the 16kHz mono capture rate VAD/STT need directly,
    without erroring ("Invalid sample rate") or needing a resampler.
    Falls back to the system default input device (None) if no DirectSound
    headset mic is found.
    """
    pa = pyaudio.PyAudio()
    try:
        ds_index = pa.get_host_api_info_by_type(pyaudio.paDirectSound)["index"]
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if (
                info["hostApi"] == ds_index
                and info["maxInputChannels"] > 0
                and "headset microphone" in info["name"].lower()
            ):
                return i
    except OSError:
        pass  # DirectSound host API not available on this system
    finally:
        pa.terminate()
    return None


async def run_local():
    """Entry point for the CLI/local-mic mode (``python bot.py --local``)."""
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            input_device_index=_find_headset_mic_device_index(),
        )
    )

    await run_bot(transport, handle_sigint=True)


def _check_port_available(host: str, port: int) -> None:
    """Exit with a clear message if `port` is already bound.

    pipecat's own WebRTC runner prints "Bot ready!" (see
    `pipecat.runner.run.main`, which calls `_print_startup_message` before
    `uvicorn.run`) BEFORE it actually attempts to bind the port - so a
    second instance started against an already-used port prints a false
    "ready" message and only fails later. Checking here, before handing
    off to pipecat's runner at all, avoids that misleading sequence.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        try:
            s.bind((host, port))
        except OSError:
            sys.exit(
                f"Port {port} on {host} is already in use - is another "
                f"instance of this bot already running?"
            )


if __name__ == "__main__":
    import asyncio

    if "--local" in sys.argv:
        asyncio.run(run_local())
    else:
        from pipecat.runner.run import RUNNER_HOST, RUNNER_PORT, main

        # Mirrors pipecat's own --host/--port argparse defaults so this
        # check targets the same address the runner will actually bind.
        host = sys.argv[sys.argv.index("--host") + 1] if "--host" in sys.argv else RUNNER_HOST
        port = (
            int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else RUNNER_PORT
        )
        _check_port_available(host, port)

        main()
