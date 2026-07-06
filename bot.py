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

import os
import sys
import time

import pyaudio
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import ErrorFrame, LLMRunFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments, SmallWebRTCRunnerArguments
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.groq.stt import GroqSTTService
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
logger.add(sys.stderr, level="CONVO")

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant in a voice conversation. Your responses will "
    "be spoken aloud, so avoid emojis, bullet points, or other formatting that "
    "can't be spoken. "
    "You understand all languages. Always respond in English regardless of what language the user speaks in. "
    "Keep every response short and direct — one or two "
    "sentences by default, never more than a few. No filler, no preamble, no "
    "restating the question, no hedging caveats. Answer exactly what was asked "
    "and stop."
)

FALLBACK_ERROR_MESSAGE = "Sorry, I hit a glitch there. Could you say that again?"
FALLBACK_COOLDOWN_SECS = 5.0

KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_heart")


class MultilingualGroqSTTService(GroqSTTService):
    """GroqSTTService with Whisper language auto-detection enabled.

    The upstream service hardcodes `language=EN` and always sends it to the
    API, which causes Whisper to force-fit every utterance into English
    phonemes.  Omitting the language parameter entirely tells Whisper to
    detect the language from the audio — works for Urdu, English, and
    code-switched input without any extra configuration.

    NOTE: The TTS side (KokoroTTSService) does not support Urdu, so the bot
    can understand Urdu input and the LLM will reply in Urdu, but the spoken
    output will be garbled or silent for Urdu text.  Fix: swap KokoroTTSService
    for Google Cloud TTS (ur-IN) or Azure TTS (ur-PK) when Urdu voice output
    is needed.
    """

    async def _transcribe(self, audio: bytes) -> Transcription:
        kwargs = {
            "file": ("audio.wav", audio, "audio/wav"),
            "model": self._settings.model,
            "response_format": "verbose_json" if self._include_prob_metrics else "json",
            "language": "ur",
        }
        if self._settings.prompt is not None:
            kwargs["prompt"] = self._settings.prompt
        if self._settings.temperature is not None:
            kwargs["temperature"] = self._settings.temperature
        result = await self._client.audio.transcriptions.create(**kwargs)
        # repr() is ASCII-safe: shows \uXXXX escapes for non-Latin chars so
        # you can tell whether Whisper returned actual Urdu Unicode or English.
        logger.debug(f"STT raw transcript: {repr(result.text)}")
        return result


async def run_bot(transport: BaseTransport, *, handle_sigint: bool = False):
    stt = MultilingualGroqSTTService(
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
        ),
    )

    context = LLMContext()
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

    context.add_message({"role": "developer", "content": "Please introduce yourself to the user."})
    await worker.queue_frames([LLMRunFrame()])

    runner = WorkerRunner(handle_sigint=handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Entry point used by the WebRTC dev runner (``python bot.py -t webrtc``)."""
    if not isinstance(runner_args, SmallWebRTCRunnerArguments):
        raise RuntimeError("This bot only supports the WebRTC transport (run with -t webrtc).")

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
        pass  # WASAPI not available on this system
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
