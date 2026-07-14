"""
Regenerates fallback_audio.wav - the pre-recorded apology played when
Kokoro TTS itself is broken (corrupted model, crash, resource exhaustion)
and can't be used to speak a fallback apology through itself. See
FALLBACK_AUDIO / on_pipeline_error in bot.py for how it's used.

The spoken text here MUST match bot.py's FALLBACK_ERROR_MESSAGE constant -
they're pushed together (TTSTextFrame + this audio's samples) so a
client/transcript sees consistent text and audio. If you change one,
regenerate to match the other.

Run whenever FALLBACK_ERROR_MESSAGE changes, or to pick a different voice:
    python scripts/generate_fallback_audio.py
"""
import os
import wave

import numpy as np
from kokoro_onnx import Kokoro

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(REPO_ROOT, "fallback_audio.wav")

# Kept in sync with bot.py's FALLBACK_ERROR_MESSAGE by hand rather than
# imported from bot.py, since importing bot.py here would require a real
# GROQ_API_KEY just to generate an audio file.
FALLBACK_ERROR_MESSAGE = "Sorry, I hit a glitch there. Could you say that again?"
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_heart")


def main():
    model_path = os.path.expanduser(r"~\.cache\pipecat\kokoro-onnx\kokoro-v1.0.onnx")
    voices_path = os.path.expanduser(r"~\.cache\pipecat\kokoro-onnx\voices-v1.0.bin")
    kokoro = Kokoro(model_path, voices_path)

    samples, sample_rate = kokoro.create(
        FALLBACK_ERROR_MESSAGE, voice=KOKORO_VOICE, speed=1.0, lang="en-us"
    )
    pcm16 = (samples * 32767).astype(np.int16)

    with wave.open(OUTPUT_PATH, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())

    print(f"Wrote {OUTPUT_PATH} ({len(samples) / sample_rate:.2f}s, {sample_rate}Hz)")


if __name__ == "__main__":
    main()
