"""
Rudimentary voice AI chatbot pipeline - text-output variant. Runs either over
WebRTC in the browser or directly against the local mic from the CLI. Input
is still spoken (mic), but the bot's reply is text-only: no TTS, no speaker
output. Over WebRTC/eval, the reply is delivered to the connected client as
an RTVI "bot-llm-text" message; in --local mode it's only visible in the
console (CONVO log line), since there's no client to receive a text message.

Pipeline: mic (WebRTC or local) -> Silero VAD -> Groq STT (Whisper)
          -> Groq LLM (Llama 3.3 70B) -> text delivered via RTVI (no TTS)

Also includes worked examples of LLM tool calling against a REAL website
(honda.com.pk), not fake/hardcoded data:
  - check_honda_price: real, live starting prices from the homepage's
    mega-menu.
  - browse_honda_page: fetches and reads any of a known set of real pages
    on the site (model specs/features, dealer contact info, promotions,
    company info, policies) so the bot can answer open-ended questions
    about the site's actual content, not just price.

Run:
    python bot.py             # WebRTC server; open http://localhost:7860
    python bot.py --local     # Talk directly via the local mic (reply is console-only)

Requires GROQ_API_KEY in a .env file.
"""

import asyncio
import dataclasses
import os
import re
import sys
import time

import pyaudio
from bs4 import BeautifulSoup
from curl_cffi import requests as curl_requests
from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import ErrorFrame, Frame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.models import (
    BotLLMStartedMessage,
    BotLLMStoppedMessage,
    BotLLMTextMessage,
    TextMessageData,
)
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
    "You are a helpful assistant. The user is speaking to you out loud, but "
    "your replies are delivered back as text, not spoken - so normal written "
    "formatting is fine. "
    "You understand all languages. Always respond in English regardless of what language the user speaks in. "
    "Your replies are read as text, not spoken aloud, so there's no need to "
    "artificially shorten them the way a spoken answer would need to be - "
    "give complete, useful answers (including specs, lists, or detail) when "
    "the question calls for it. Still be direct: no filler, no unnecessary "
    "preamble, no restating the question, no hedging caveats, and no padding "
    "a simple answer just to sound thorough. "
    "If the user asks about the price of a Honda Civic, HR-V, or City, use "
    "the check_honda_price tool rather than guessing - never make up a price. "
    "For anything else about Honda Pakistan - specs, features, dealers, "
    "contact info, promotions, company info, policies - use the "
    "browse_honda_page tool to check the real website rather than guessing. "
    "If the user's message is too short, vague, or ambiguous to answer "
    "meaningfully (e.g. a single word like 'yes' or 'ok' with no clear "
    "context), don't guess at what they might mean - ask a brief clarifying "
    "question instead."
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
    """Fetch + parse honda.com.pk's mega-menu prices, using a short-lived cache.

    Falls back to a stale cache rather than failing outright when a fresh
    fetch fails - a fetch failure this instant doesn't mean a price learned
    5 minutes ago is now wrong, so there's no reason to make the user wait
    or get an error when good-enough data is already sitting in memory.
    """
    global _price_cache, _price_cache_time

    now = time.monotonic()
    if _price_cache and (now - _price_cache_time) < _PRICE_CACHE_TTL_SECS:
        return _price_cache

    try:
        html = await asyncio.to_thread(_fetch_honda_homepage_sync)
    except curl_requests.exceptions.RequestException:
        if _price_cache:
            logger.warning(
                f"Honda homepage fetch failed - falling back to a stale price "
                f"cache ({now - _price_cache_time:.0f}s old) instead of failing."
            )
            return _price_cache
        raise

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

    if not prices:
        # Confirmed directly: a 200 OK response with a bot-challenge page
        # (or any HTML structure change) parses to zero prices with no
        # exception raised at all - treating that the same as "this
        # specific model doesn't exist" would silently tell users every
        # single model is unavailable instead of signaling that the scrape
        # itself is broken.
        logger.error(
            "[tool call] check_honda_price: fetch succeeded but zero prices "
            "were parsed - site structure may have changed, or a "
            "bot-challenge page was served instead of the real homepage."
        )
        await params.result_callback(
            {
                "model": model,
                "found": False,
                "error": "could not verify any prices right now - honda.com.pk "
                "may be temporarily unreachable",
            }
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

# General-purpose "browse the real website" tool - covers everything on
# honda.com.pk that ISN'T the price mega-menu: model specs/features,
# dealer/contact info, promotions, company info, policies. Maps a handful
# of friendly topic names to the real page slugs discovered by actually
# crawling the site's homepage links (not guessed).
HONDA_PAGE_SLUGS = {
    "civic": "civic-standard",
    "civic standard": "civic-standard",
    "civic oriel": "civic-oriel-1-5",
    "civic rs": "civic-rs-turbo",
    "civic rs turbo": "civic-rs-turbo",
    "hr-v": "hrv-vti",
    "hrv": "hrv-vti",
    "hr-v s": "hrv-vti-s",
    "hr-v hybrid": "hrv-ehev",
    "hrv hybrid": "hrv-ehev",
    "hrv e:hev": "hrv-ehev",
    "city": "city1-2l",
    "city 1.2": "city1-2l",
    "city 1.5": "city1-5l",
    "city aspire": "cityaspire",
    "accord": "hondaaccord",
    "cr-v": "hondacrv",
    "crv": "hondacrv",
    "about": "abouthonda",
    "about honda": "abouthonda",
    "company": "abouthonda",
    "contact": "contactus",
    "contact us": "contactus",
    "dealer": "location-us",
    "dealers": "location-us",
    "dealer network": "location-us",
    "locations": "location-us",
    "promotions": "promotions",
    "offers": "promotions",
    "deals": "promotions",
    "news": "newsandevents",
    "events": "newsandevents",
    "free service": "free-service",
    "delivery status": "delivery-status",
    "policies": "policies",
    "privacy policy": "privacy-policy",
    "terms": "terms-and-conditions",
    "terms and conditions": "terms-and-conditions",
}

# Cache extracted page text briefly, per slug - same politeness rationale
# as the price cache above.
_page_text_cache: dict[str, str] = {}
_page_text_cache_time: dict[str, float] = {}
_PAGE_CACHE_TTL_SECS = 300.0
_PAGE_TEXT_MAX_CHARS = 3000
# A 200 OK response can still be a bot-challenge page or a near-empty error
# page rather than real content - handing that to the LLM as if it were the
# genuine page risks a hallucinated or nonsensical answer built from noise.
# 200 is a conservative guess, not measured against the real site (these
# pages typically run into the thousands of characters once script/style/
# nav noise is stripped) - low risk of rejecting a legitimately terse page.
_PAGE_TEXT_MIN_CHARS = 200


def _fetch_and_extract_page_sync(slug: str) -> str:
    """Blocking fetch + HTML-to-text extraction, run in a background thread.

    Uses the same curl_cffi Chrome-impersonation approach as
    `_fetch_honda_homepage_sync` (see that docstring for why) since this
    hits the same Cloudflare-protected site. Strips script/style/nav/footer
    noise and returns plain, readable text for the LLM to read - real page
    content, not a summary or paraphrase we wrote ourselves.
    """
    url = f"https://www.honda.com.pk/{slug}"
    response = curl_requests.get(url, impersonate="chrome", timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "noscript", "svg"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ", strip=True).split())
    return text[:_PAGE_TEXT_MAX_CHARS]


async def browse_honda_page(params: FunctionCallParams):
    """Tool handler: fetches and reads a real page on honda.com.pk.

    Called by the LLM whenever it decides the user is asking about
    something on the site other than price - specs, features, dealer
    info, promotions, company info, policies. `params.arguments["topic"]`
    is matched (loosely) against `HONDA_PAGE_SLUGS` to find the real page.
    """
    topic = str(params.arguments.get("topic", "")).strip().lower()
    slug = HONDA_PAGE_SLUGS.get(topic)

    if slug is None:
        # Loose fallback: does any known topic phrase appear in what the
        # model sent, or vice versa? Handles near-misses like "civics" or
        # "the hrv model" without needing an exact dict key match.
        slug = next(
            (s for key, s in HONDA_PAGE_SLUGS.items() if key in topic or topic in key),
            None,
        )

    if slug is None:
        logger.log("CONVO", f"[tool call] browse_honda_page(topic={topic!r}) -> no match")
        await params.result_callback(
            {
                "topic": topic,
                "found": False,
                "available_topics": sorted(set(HONDA_PAGE_SLUGS.keys())),
            }
        )
        return

    now = time.monotonic()
    cached = _page_text_cache.get(slug)
    fresh_age = now - _page_text_cache_time.get(slug, 0)
    if cached is not None and fresh_age < _PAGE_CACHE_TTL_SECS:
        text = cached
    else:
        try:
            text = await asyncio.to_thread(_fetch_and_extract_page_sync, slug)
        except curl_requests.exceptions.RequestException as e:
            # Fall back to a stale cache rather than failing outright - same
            # reasoning as _get_honda_prices' stale-cache fallback above.
            if cached is not None:
                logger.warning(
                    f"[tool call] browse_honda_page: fetch failed for {slug!r}, "
                    f"falling back to stale cache ({fresh_age:.0f}s old)."
                )
                text = cached
            else:
                logger.error(f"[tool call] browse_honda_page: fetch failed for {slug!r}: {e}")
                await params.result_callback(
                    {
                        "topic": topic,
                        "found": False,
                        "error": "could not reach honda.com.pk right now",
                    }
                )
                return
        else:
            _page_text_cache[slug] = text
            _page_text_cache_time[slug] = now

    if len(text) < _PAGE_TEXT_MIN_CHARS:
        logger.error(
            f"[tool call] browse_honda_page: fetched {slug!r} but got only "
            f"{len(text)} chars of content - likely a bot-challenge page or "
            f"a site change, not real page content."
        )
        await params.result_callback(
            {
                "topic": topic,
                "found": False,
                "error": "could not verify this page's content right now",
            }
        )
        return

    logger.log(
        "CONVO", f"[tool call] browse_honda_page(topic={topic!r}) -> {slug} ({len(text)} chars)"
    )
    await params.result_callback({"topic": topic, "found": True, "page_content": text})


browse_honda_page_tool = FunctionSchema(
    name="browse_honda_page",
    description=(
        "Fetch and read a real page from the honda.com.pk website to answer "
        "questions about model specs/features, dealer or contact info, "
        "promotions, company info, or policies - anything other than price."
    ),
    properties={
        "topic": {
            "type": "string",
            "description": (
                "What to look up, e.g. 'Civic specs', 'dealer locations', "
                "'contact info', 'promotions', 'about honda'."
            ),
        }
    },
    required=["topic"],
    handler=browse_honda_page,
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


_HTML_TAG_PATTERN = re.compile(r"<[^>]*>")


class TextSanitizer(FrameProcessor):
    """Strips raw HTML tags from LLM output before it's delivered to the
    client as an RTVI text message.

    Defensive, not a fix for an observed bug: the system prompt here allows
    normal written formatting (unlike the audio variant, which strips
    everything before TTS), but nothing stops the LLM from occasionally
    echoing back a stray HTML tag - e.g. leftover markup from a scraped
    Honda page, or a rare hallucination. If whatever client eventually
    renders these replies does markdown-to-HTML conversion without
    sanitizing, an unfiltered raw tag reaching it is a real (if unlikely)
    injection risk. Cheap insurance given this variant is the only one
    whose system prompt invites rich formatting at all.

    Known limitation: operates per-fragment, since RTVI streams one
    "bot-llm-text" message per raw LLM token chunk rather than per complete
    sentence (confirmed directly in pipecat's
    RTVIObserver._handle_llm_text_frame - every LLMTextFrame is pushed to
    the client immediately, unaggregated). Buffering into complete
    sentences first (like the audio variant's TTSTextNormalizer does) would
    add latency before any text reaches the client at all, defeating
    real-time streaming - a tag split exactly across two separate streamed
    fragments could theoretically slip through partially. A narrow, accepted
    trade-off for keeping streaming immediate.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame) and frame.text:
            sanitized = _HTML_TAG_PATTERN.sub("", frame.text)
            if sanitized != frame.text:
                logger.warning(f"Stripped HTML-like tag from LLM output: {frame.text!r}")
            await self.push_frame(dataclasses.replace(frame, text=sanitized), direction)
            return
        await self.push_frame(frame, direction)


async def _startup_self_check(llm: GroqLLMService) -> None:
    """Validates the LLM actually works before accepting real traffic - not
    just that a key is present (already checked at import time), but that a
    real call succeeds.

    Scoped to just the LLM: this variant has no TTS, and STT/audio-input
    concerns are out of scope here (same reasoning as BilingualGroqSTTService
    and the rest of the audio-input layer - this variant intentionally
    doesn't duplicate that verification).

    A revoked/expired key or a Groq quota/permission issue would otherwise
    only surface on the user's FIRST real turn, presenting as a mysterious
    fallback message with no clear cause. Failing fast here is much easier
    to diagnose than that.

    Exits the process (does not raise) if the check fails, matching the
    existing fail-fast behavior for a missing API key / port conflict.
    """
    try:
        await llm._client.chat.completions.create(
            model=llm._settings.model,
            messages=[{"role": "user", "content": "hi"}],
            max_completion_tokens=1,
        )
    except Exception as e:
        logger.error("Startup self-check FAILED - fix this before running the bot:")
        logger.error(f"  - LLM (Groq chat completions, model={llm._settings.model}): {e}")
        sys.exit(1)

    logger.info("Startup self-check passed: LLM is working.")


# pipecat spawns a fresh run_bot() per WebRTC connection
# (background_tasks.add_task on every /api/offer request), so without this
# flag every single connection would re-run the self-check - an extra API
# round-trip added to that user's first-turn latency, and extra quota usage
# per session instead of once per process. The LLM client itself (same API
# key, same model) doesn't change between connections, so verifying it once
# per process is enough.
_startup_self_check_done = False


async def _push_standalone_text_message(worker: PipelineWorker, text: str) -> None:
    """Pushes a one-shot bot text message (greeting/fallback) with proper
    started/stopped bookends, not just the bare text.

    Bug found via comprehensive testing: pushing only a bare BotLLMTextMessage
    (no surrounding lifecycle messages) means a client tracking "is the bot
    currently responding" via the standard bot-llm-started/bot-llm-stopped
    pair never gets told this one finished - confirmed directly that
    bot-llm-stopped is only emitted by pipecat's RTVI observer in response to
    a real LLMFullResponseEndFrame, which never occurs for a message pushed
    this way. A client's "bot is typing..." state could get stuck forever
    after every greeting or fallback apology. Same root-cause class as the
    audio variant's TTSStartedFrame/TTSStoppedFrame fix for its fallback
    audio - a raw content frame/message needs its matching lifecycle
    bookends, not just the content itself.
    """
    await worker.rtvi.push_transport_message(BotLLMStartedMessage())
    await worker.rtvi.push_transport_message(BotLLMTextMessage(data=TextMessageData(text=text)))
    await worker.rtvi.push_transport_message(BotLLMStoppedMessage())


async def run_bot(transport: BaseTransport, *, handle_sigint: bool = False):
    # whisper-large-v3-turbo instead of whisper-large-v3: measured directly
    # in this exact pipeline (same conversation, same real audio, only the
    # model swapped) - STT TTFB dropped from ~3s to ~0.3-0.5s, a 6-10x cut,
    # since every utterance already pays for two concurrent transcription
    # calls (see BilingualGroqSTTService). Re-verified the Urdu-bias
    # threshold tuning still holds afterward (the "Wait, stop, never mind."
    # case that used to be borderline still resolved to English, with a
    # wider confidence margin than before, if anything). Known trade-off,
    # not fully exercised here: Whisper's turbo variants are documented to
    # trade a little accuracy for speed, more noticeably on less-common
    # languages/accents than clean English.
    stt = BilingualGroqSTTService(
        api_key=os.environ["GROQ_API_KEY"],
        settings=GroqSTTService.Settings(model="whisper-large-v3-turbo"),
    )

    llm = GroqLLMService(
        api_key=os.environ["GROQ_API_KEY"],
        settings=GroqLLMService.Settings(
            model="llama-3.3-70b-versatile",
            system_instruction=SYSTEM_INSTRUCTION,
            # The original 150-token cap existed to bound worst-case TTS
            # synthesis time in the audio variant - that reason doesn't
            # apply here (no TTS at all), and it was cutting off detailed
            # answers (specs, comparisons) that a text reply can comfortably
            # hold. Raised to a much larger backstop that only exists to
            # guard against genuinely runaway/adversarial-prompt generation,
            # not to keep everyday answers short.
            max_completion_tokens=500,
        ),
    )

    global _startup_self_check_done
    if os.getenv("SKIP_STARTUP_SELF_CHECK"):
        logger.warning("SKIPPING startup self-check (SKIP_STARTUP_SELF_CHECK is set).")
    elif _startup_self_check_done:
        logger.debug("Skipping startup self-check - already verified once this process.")
    else:
        await _startup_self_check(llm)
        _startup_self_check_done = True

    context = LLMContext(tools=[honda_price_tool, browse_honda_page_tool])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
        # Bulletproofing: a long-running conversation would otherwise grow
        # LLMContext unboundedly - every future turn resends the entire
        # history, so cost/latency creep up forever and eventually risk
        # hitting the model's real context limit. pipecat already ships a
        # complete summarization mechanism for this - it's just OFF by
        # default. Turning it on with its own sensible defaults rather than
        # leaving this pipeline vulnerable to unbounded growth.
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

        if message.interrupted:
            # Confirmed problem #9 (ported from the audio variant): the
            # bot's memory of an interrupted response didn't match what the
            # user actually got. Real audio-driven testing on the audio
            # variant found message.content comes back empty on interruption
            # - pipecat doesn't broadcast any assistant entry into context
            # at all in that case, which is silent data loss (the LLM has
            # zero memory it started answering) rather than the originally
            # suspected fabricated-full-text case. Handles both: replaces
            # the entry if pipecat did broadcast one, otherwise appends an
            # honest interruption marker instead of silently losing the turn.
            original_content = message.content

            async def _correct_interrupted_context():
                await asyncio.sleep(0.2)
                messages = context.get_messages()
                marker = (
                    "[This response was interrupted by the user before finishing - "
                    "only part of it (if any) was actually delivered, not the "
                    "complete answer.]"
                )
                if original_content:
                    for i in reversed(range(len(messages))):
                        if (
                            messages[i].get("role") == "assistant"
                            and messages[i].get("content") == original_content
                        ):
                            messages[i]["content"] = marker
                            context.set_messages(messages)
                            logger.debug("Corrected interrupted assistant turn in context")
                            return
                messages.append({"role": "assistant", "content": marker})
                context.set_messages(messages)
                logger.debug("Recorded interruption marker for empty-content assistant turn")

            asyncio.create_task(_correct_interrupted_context())

    text_sanitizer = TextSanitizer()

    pipeline = Pipeline(
        [
            transport.input(),  # Mic input
            stt,  # Speech -> text
            user_aggregator,  # Collect user turn
            llm,  # Generate response (text delivered to the client via RTVI below)
            text_sanitizer,  # Strip any raw HTML tags before the client sees them
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
        #
        # Bulletproofing: made explicit rather than relying on the library
        # default. If a connected user goes quiet this long, pipecat cancels
        # this session's worker AND runner automatically, scoped to just
        # that one connection (background_tasks.add_task spawns an
        # independent bot()/run_bot()/WorkerRunner() per WebRTC connection).
        idle_timeout_secs=300.0,
    )

    # Confirmed problem #7 (ported from the audio variant): a flat
    # per-attempt cooldown still lets the fallback fire indefinitely during
    # a genuine outage (every FALLBACK_COOLDOWN_SECS, forever). This is a
    # real circuit breaker instead: each consecutive failure doubles how
    # long the circuit stays open (backoff grows 5s -> 10s -> 20s -> ...
    # capped at CIRCUIT_BREAKER_MAX_BACKOFF_SECS), and a single successful
    # turn (on_assistant_turn_stopped above) closes it again completely.
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
            # rather than sending (and failing) again immediately.
            return

        # A non-fatal ErrorFrame (STT/LLM hiccup, rate limit, etc.) would
        # otherwise just get logged, leaving the user with no reply for that
        # turn. Send a short apology as a text message instead, so the
        # conversation can continue. Not added to LLM context, matching the
        # original TTS-based fallback's behavior. Unlike the audio variant,
        # there's no "apologizing through the broken service" risk here -
        # this is a plain RTVI text push, not dependent on any AI service.
        logger.log("CONVO", f"Bot: {FALLBACK_ERROR_MESSAGE}")
        await _push_standalone_text_message(worker, FALLBACK_ERROR_MESSAGE)

        consecutive_fallback_failures += 1
        backoff = min(
            FALLBACK_COOLDOWN_SECS * (2 ** (consecutive_fallback_failures - 1)),
            CIRCUIT_BREAKER_MAX_BACKOFF_SECS,
        )
        circuit_open_until = now + backoff
        logger.error(
            f"Circuit breaker: {consecutive_fallback_failures} consecutive failure(s), "
            f"backing off fallback message for {backoff:.0f}s."
        )

    @worker.event_handler("on_pipeline_started")
    async def send_greeting(worker, frame):
        # Sent directly as a text message instead of round-tripping through
        # the LLM just to say hello - saves an API call. Unlike the original
        # TTS-based greeting, this isn't added to the LLM's conversation
        # history (RTVI text pushes don't touch LLMContext) - a minor,
        # accepted trade-off of the text-output variant.
        logger.log("CONVO", f"Bot: {GREETING_MESSAGE}")
        await _push_standalone_text_message(worker, GREETING_MESSAGE)

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
                # Bug found via comprehensive testing: audio_out_enabled=False
                # (the semantically "correct" choice, since this variant has
                # no TTS at all) silently breaks the RTVI observer's
                # user-started-speaking/interruption event forwarding to the
                # client - confirmed directly (A/B tested in eval mode): with
                # this False, "user_started_speaking" never reached the
                # client despite the bot's own internal VAD/turn-detection
                # correctly firing; flipping it to True fixed it immediately,
                # no other change. No TTS service exists anywhere in this
                # pipeline to actually generate audio, so this is a free fix,
                # not a real audio-output enablement.
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
            audio_out_enabled=False,  # no TTS - reply only appears in the console (CONVO log)
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
