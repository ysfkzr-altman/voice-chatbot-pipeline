"""
Rudimentary voice AI chatbot pipeline. Runs either over WebRTC in the browser
or directly against the local mic/speakers from the CLI.

Pipeline: mic (WebRTC or local) -> Silero VAD -> Groq STT (Whisper)
          -> Groq LLM (Llama 3.3 70B) -> Kokoro TTS (local, no API key) -> speakers

This branch adds one worked example of LLM tool calling: a fake dealership
`check_inventory` tool (see INVENTORY / check_inventory below), so the bot
can answer "do you have a Civic in stock?" from real(ish) data instead of
just chatting generically. Swap the fake dict + lookup for a real API/DB
call to point this at an actual business's data.

Run:
    python bot.py             # WebRTC server; open http://localhost:7860
    python bot.py --local     # Talk directly via the local mic/speakers

Requires GROQ_API_KEY in a .env file.
Kokoro model files (~87 MB) are downloaded automatically on first run to
~/.cache/pipecat/kokoro-onnx/.
"""

import asyncio
import os
import sys
import time

import pyaudio
from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import ErrorFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import EvalRunnerArguments, RunnerArguments, SmallWebRTCRunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.kokoro.tts import KokoroTTSService
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
    "You are a helpful assistant for a car dealership, in a voice conversation. "
    "Your responses will be spoken aloud, so avoid emojis, bullet points, or "
    "other formatting that can't be spoken. "
    "You understand all languages. Always respond in English regardless of what language the user speaks in. "
    "Keep every response short and direct — one or two "
    "sentences by default, never more than a few. No filler, no preamble, no "
    "restating the question, no hedging caveats. Answer exactly what was asked "
    "and stop. "
    "If the user asks whether a specific car model is in stock, use the "
    "check_inventory tool rather than guessing - never make up stock numbers."
)

GREETING_MESSAGE = "Hi! How can I help you today?"
FALLBACK_ERROR_MESSAGE = "Sorry, I hit a glitch there. Could you say that again?"
FALLBACK_COOLDOWN_SECS = 5.0

KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_heart")

# Worked example of tool calling: a fake dealership inventory. In a real
# integration, check_inventory would call that business's actual
# inventory API/DB instead of reading this hardcoded dict.
INVENTORY = {
    "civic": 3,
    "accord": 0,
    "cr-v": 5,
    "mg zs": 2,
    "mg hs": 0,
}


async def check_inventory(params: FunctionCallParams):
    """Tool handler: looks up how many units of a car model are in stock.

    Called by the LLM (not directly by us) whenever it decides the user is
    asking about stock for a specific model. `params.arguments` holds
    whatever arguments the model filled in, matching the `properties`
    declared in `inventory_tool` below.
    """
    model = str(params.arguments.get("model", "")).strip().lower()
    count = INVENTORY.get(model)

    logger.log("CONVO", f"[tool call] check_inventory(model={model!r}) -> {count}")

    if count is None:
        result = {"model": model, "found": False}
    else:
        result = {"model": model, "found": True, "in_stock": count > 0, "count": count}

    # Hands the result back to the LLM, which then generates the spoken
    # reply using it - the tool call never speaks directly.
    await params.result_callback(result)


inventory_tool = FunctionSchema(
    name="check_inventory",
    description=(
        "Check how many units of a specific car model are currently in "
        "stock at the dealership."
    ),
    properties={
        "model": {
            "type": "string",
            "description": "The car model to check, e.g. 'Civic' or 'MG ZS'.",
        }
    },
    required=["model"],
    handler=check_inventory,
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

    tts = KokoroTTSService(
        settings=KokoroTTSService.Settings(voice=KOKORO_VOICE),
    )

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

    context = LLMContext(tools=[inventory_tool])
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
            llm,  # Generate response
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
    )

    last_fallback_speak_time = 0.0

    @worker.event_handler("on_pipeline_error")
    async def on_pipeline_error(worker, frame: ErrorFrame):
        nonlocal last_fallback_speak_time
        logger.error(f"Pipeline error from {frame.processor}: {frame.error}")
        if frame.fatal:
            return

        # If TTS itself is the thing failing, speaking a fallback apology
        # through that same broken TTS just triggers another ErrorFrame,
        # which re-triggers this handler - an infinite retry loop hammering
        # the TTS API. Log only; there's no way to speak our way out of a
        # dead TTS connection.
        if frame.processor is tts:
            logger.error("TTS itself is failing - not attempting a spoken fallback (would loop).")
            return

        # General safety net: never fire the spoken fallback more than once
        # every FALLBACK_COOLDOWN_SECS, regardless of source, in case some
        # other repeat-failure pattern (e.g. STT down on every turn) shows up.
        now = time.monotonic()
        if now - last_fallback_speak_time < FALLBACK_COOLDOWN_SECS:
            return
        last_fallback_speak_time = now

        # A non-fatal ErrorFrame (STT/LLM/TTS hiccup, rate limit, etc.) would
        # otherwise just get logged, leaving the user hearing silence for
        # that turn. Speak a short apology instead so the conversation can
        # continue. append_to_context=False keeps it out of the LLM history.
        await worker.queue_frames(
            [TTSSpeakFrame(text=FALLBACK_ERROR_MESSAGE, append_to_context=False)]
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


if __name__ == "__main__":
    import asyncio

    if "--local" in sys.argv:
        asyncio.run(run_local())
    else:
        from pipecat.runner.run import main

        main()
