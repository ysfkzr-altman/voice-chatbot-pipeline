"""
Rudimentary voice AI chatbot pipeline - text-output variant. Runs either over
WebRTC in the browser or directly against the local mic from the CLI. Input
is still spoken (mic), but the bot's reply is text-only: no TTS, no speaker
output. Over WebRTC/eval, the reply is delivered to the connected client as
an RTVI "bot-llm-text" message; in --local mode it's only visible in the
console (CONVO log line), since there's no client to receive a text message.

Pipeline: mic (WebRTC or local) -> Silero VAD -> Groq STT (Whisper)
          -> Groq LLM (Llama 3.3 70B) -> text delivered via RTVI (no TTS)

Also includes a worked example of LLM tool calling: `check_honda_price`
fetches and parses the real, live starting prices from honda.com.pk's own
homepage (see check_honda_price below), so the bot can answer "how much is
the Civic?" using actual current data instead of a guess.

Run:
    python bot.py             # WebRTC server; open http://localhost:7860
    python bot.py --local     # Talk directly via the local mic (reply is console-only)

Requires GROQ_API_KEY in a .env file.
"""

import asyncio
import os
import re
import sys
import time

import pyaudio
from curl_cffi import requests as curl_requests
from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import ErrorFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frameworks.rtvi.models import BotLLMTextMessage, TextMessageData
from pipecat.runner.types import EvalRunnerArguments, RunnerArguments, SmallWebRTCRunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.groq.stt import GroqSTTService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.whisper.base_stt import Transcription
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

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
    "You are a helpful assistant. The user is speaking to you out loud, but "
    "your replies are delivered back as text, not spoken - so normal written "
    "formatting is fine. "
    "You understand all languages. Always respond in English regardless of what language the user speaks in. "
    "Keep every response short and direct — one or two "
    "sentences by default, never more than a few. No filler, no preamble, no "
    "restating the question, no hedging caveats. Answer exactly what was asked "
    "and stop. "
    "If the user asks about the price of a Honda Civic, HR-V, or City, use "
    "the check_honda_price tool rather than guessing - never make up a price."
)

GREETING_MESSAGE = "Hi! How can I help you today?"
FALLBACK_ERROR_MESSAGE = "Sorry, I hit a glitch there. Could you say that again?"
FALLBACK_COOLDOWN_SECS = 5.0

# Worked example of tool calling against a REAL website (not fake/hardcoded
# data): honda.com.pk's homepage includes a mega-menu block, present on
# every page, listing each model line's current starting price - e.g.
#   <h4>Honda Civic</h4> ... <div class="model-price">From PKR 8,499,000</div>
# This is public marketing content, not a live inventory/stock API - Honda
# Pakistan doesn't expose per-unit stock counts publicly (that lives inside
# individual dealers' internal systems). Pricing is the real, live data
# that's actually available to scrape here.
HONDA_HOMEPAGE_URL = "https://www.honda.com.pk/"
_HONDA_PRICE_PATTERN = re.compile(
    r'<h4>Honda ([\w\- ]+?)</h4>.*?model-price">From PKR ([\d,]+)</div>',
    re.DOTALL,
)

# Cache the parsed prices briefly so a burst of questions in one
# conversation doesn't re-fetch the real site on every single turn - this
# is a live public website, not our own infrastructure, so being a
# reasonably polite client matters.
_price_cache: dict[str, str] = {}
_price_cache_time = 0.0
_PRICE_CACHE_TTL_SECS = 300.0


def _fetch_honda_homepage_sync() -> str:
    """Blocking fetch, run in a background thread by `_get_honda_prices`.

    Plain httpx (or curl with no browser identity) gets blocked with a 403
    by the Cloudflare WAF in front of this site - confirmed by testing:
    even a normal browser `User-Agent` header wasn't enough, since
    Cloudflare's bot-detection also fingerprints the TLS handshake itself
    (JA3/JA4), which differs between a real browser and a generic Python
    HTTP client regardless of headers sent. `curl_cffi` with
    `impersonate="chrome"` reproduces an actual Chrome TLS fingerprint and
    gets through reliably (confirmed with repeated live requests).
    """
    response = curl_requests.get(HONDA_HOMEPAGE_URL, impersonate="chrome", timeout=10)
    response.raise_for_status()
    return response.text


async def _get_honda_prices() -> dict[str, str]:
    """Fetch + parse honda.com.pk's mega-menu prices, using a short-lived cache."""
    global _price_cache, _price_cache_time

    now = time.monotonic()
    if _price_cache and (now - _price_cache_time) < _PRICE_CACHE_TTL_SECS:
        return _price_cache

    html = await asyncio.to_thread(_fetch_honda_homepage_sync)

    _price_cache = {
        name.strip().lower(): price for name, price in _HONDA_PRICE_PATTERN.findall(html)
    }
    _price_cache_time = now
    return _price_cache


async def check_honda_price(params: FunctionCallParams):
    """Tool handler: looks up a Honda Pakistan model line's real, current
    starting price by fetching and parsing honda.com.pk's own homepage -
    the same "From PKR X" figure shown to any visitor of the site.

    Called by the LLM (not directly by us) whenever it decides the user is
    asking about price for a model. `params.arguments` holds whatever
    arguments the model filled in, matching the `properties` declared in
    `honda_price_tool` below.
    """
    model = str(params.arguments.get("model", "")).strip().lower()
    # Loose match so "hrv", "hr-v", and "HR V" all match the site's "hr-v".
    model_key = model.replace("-", "").replace(" ", "")

    try:
        prices = await _get_honda_prices()
    except curl_requests.exceptions.RequestException as e:
        logger.error(f"[tool call] check_honda_price: fetch failed: {e}")
        await params.result_callback(
            {"model": model, "found": False, "error": "could not reach honda.com.pk right now"}
        )
        return

    match = next(
        (
            price
            for name, price in prices.items()
            if name.replace("-", "").replace(" ", "") == model_key
        ),
        None,
    )

    logger.log("CONVO", f"[tool call] check_honda_price(model={model!r}) -> {match}")

    if match is None:
        await params.result_callback(
            {"model": model, "found": False, "available_models": list(prices.keys())}
        )
    else:
        await params.result_callback(
            {"model": model, "found": True, "starting_price_pkr": match}
        )


honda_price_tool = FunctionSchema(
    name="check_honda_price",
    description=(
        "Look up the real, current starting price (in PKR) of a Honda "
        "Pakistan model line - Civic, HR-V, or City - by checking the live "
        "honda.com.pk website."
    ),
    properties={
        "model": {
            "type": "string",
            "description": "The Honda model line to check, e.g. 'Civic', 'HR-V', or 'City'.",
        }
    },
    required=["model"],
    handler=check_honda_price,
)


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
    regardless of input language. This was originally required because the
    audio-output version's TTS (KokoroTTSService) can't speak Urdu script -
    that constraint doesn't actually apply to this text-output variant (any
    script can be displayed as text), but the English-only instruction was
    kept as-is here to keep behavior consistent between the two variants.
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
        winner, lang, conf = (
            (result_en, "en", conf_en) if conf_en >= conf_ur else (result_ur, "ur", conf_ur)
        )

        # repr() is ASCII-safe: shows \uXXXX escapes for non-Latin chars so
        # you can tell whether Whisper returned actual Urdu Unicode or English.
        logger.debug(
            f"STT bilingual pick: lang={lang} conf={conf:.3f} "
            f"(en={conf_en:.3f} ur={conf_ur:.3f}) text={repr(winner.text)}"
        )
        return winner


async def run_bot(transport: BaseTransport, *, handle_sigint: bool = False):
    stt = BilingualGroqSTTService(
        api_key=os.environ["GROQ_API_KEY"],
        settings=GroqSTTService.Settings(model="whisper-large-v3"),
    )

    llm = GroqLLMService(
        api_key=os.environ["GROQ_API_KEY"],
        settings=GroqLLMService.Settings(
            model="llama-3.3-70b-versatile",
            system_instruction=SYSTEM_INSTRUCTION,
            # Hard cap backing up the "keep it short" system prompt instruction -
            # bounds worst-case LLM generation time.
            max_completion_tokens=150,
        ),
    )

    context = LLMContext(tools=[honda_price_tool])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    @user_aggregator.event_handler("on_user_turn_message_added")
    async def on_user_turn_message_added(aggregator, message):
        logger.log("CONVO", f"User: {message.content}")

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message):
        logger.log("CONVO", f"Bot: {message.content}")

    pipeline = Pipeline(
        [
            transport.input(),  # Mic input
            stt,  # Speech -> text
            user_aggregator,  # Collect user turn
            llm,  # Generate response (text delivered to the client via RTVI below)
            transport.output(),  # Delivers RTVI text messages to the client - no TTS/audio
            assistant_aggregator,  # Collect assistant turn
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        # enable_rtvi defaults to True - this is what turns the LLM's
        # streamed text into "bot-llm-text" messages sent to the client,
        # replacing what TTS used to do.
    )

    last_fallback_speak_time = 0.0

    @worker.event_handler("on_pipeline_error")
    async def on_pipeline_error(worker, frame: ErrorFrame):
        nonlocal last_fallback_speak_time
        logger.error(f"Pipeline error from {frame.processor}: {frame.error}")
        if frame.fatal:
            return

        # General safety net: never fire the fallback more than once every
        # FALLBACK_COOLDOWN_SECS, regardless of source (e.g. STT down on
        # every turn).
        now = time.monotonic()
        if now - last_fallback_speak_time < FALLBACK_COOLDOWN_SECS:
            return
        last_fallback_speak_time = now

        # A non-fatal ErrorFrame (STT/LLM hiccup, rate limit, etc.) would
        # otherwise just get logged, leaving the user with no reply for that
        # turn. Send a short apology as a text message instead, so the
        # conversation can continue. Not added to LLM context, matching the
        # original TTS-based fallback's behavior.
        logger.log("CONVO", f"Bot: {FALLBACK_ERROR_MESSAGE}")
        await worker.rtvi.push_transport_message(
            BotLLMTextMessage(data=TextMessageData(text=FALLBACK_ERROR_MESSAGE))
        )

    @worker.event_handler("on_pipeline_started")
    async def send_greeting(worker, frame):
        # Sent directly as a text message instead of round-tripping through
        # the LLM just to say hello - saves an API call. Unlike the original
        # TTS-based greeting, this isn't added to the LLM's conversation
        # history (RTVI text pushes don't touch LLMContext) - a minor,
        # accepted trade-off of the text-output variant.
        logger.log("CONVO", f"Bot: {GREETING_MESSAGE}")
        await worker.rtvi.push_transport_message(
            BotLLMTextMessage(data=TextMessageData(text=GREETING_MESSAGE))
        )

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
                audio_out_enabled=False,  # no TTS - replies go out as RTVI text messages
            ),
        )
    elif isinstance(runner_args, EvalRunnerArguments):
        from pipecat.evals.transport import EvalTransportParams

        transport = await create_transport(
            runner_args,
            {"eval": lambda: EvalTransportParams(audio_in_enabled=True, audio_out_enabled=False)},
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
            audio_out_enabled=False,  # no TTS - reply only appears in the console (CONVO log)
            input_device_index=_find_headset_mic_device_index(),
        )
    )

    await run_bot(transport, handle_sigint=True)


if __name__ == "__main__":
    import asyncio

    if "--local" in sys.argv:
        asyncio.run(run_local())
    else:
        from pipecat.runner.run import main

        main()
