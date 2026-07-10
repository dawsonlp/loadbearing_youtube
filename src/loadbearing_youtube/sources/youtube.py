"""YouTube transcript source.

Uses ``youtube-transcript-api`` (v1.x) for the transcript and a best-effort
oEmbed call for the title/author. No API key required for either.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request

from ..models import Segment, Transcript
from .base import TranscriptError, TranscriptSource, register_source

_ID_PATTERNS = [
    re.compile(r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/)([^&\n?#/]+)"),
    re.compile(r"youtube\.com/shorts/([^&\n?#/]+)"),
    re.compile(r"youtube\.com/live/([^&\n?#/]+)"),
]

_BARE_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(url_or_id: str) -> str:
    """Extract an 11-char video id from a URL, or accept a bare id."""
    candidate = url_or_id.strip()
    if _BARE_ID.match(candidate):
        return candidate
    for pattern in _ID_PATTERNS:
        match = pattern.search(candidate)
        if match:
            return match.group(1)
    raise TranscriptError(f"Could not extract a YouTube video id from: {url_or_id}")


def _fetch_metadata(video_id: str) -> tuple[str | None, str | None]:
    """Return ``(title, author)`` via oEmbed. Best-effort; never raises."""
    watch_url = f"https://www.youtube.com/watch?v={video_id}"
    oembed = "https://www.youtube.com/oembed?" + urllib.parse.urlencode(
        {"url": watch_url, "format": "json"}
    )
    try:
        req = urllib.request.Request(oembed, headers={"User-Agent": "loadbearing/0.1"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.load(resp)
        return data.get("title"), data.get("author_name")
    except Exception:
        return None, None


class YouTubeSource(TranscriptSource):
    name = "youtube"

    def matches(self, url: str) -> bool:
        return (
            "youtube.com" in url
            or "youtu.be" in url
            or bool(_BARE_ID.match(url.strip()))
        )

    def fetch(self, url: str, languages: list[str]) -> Transcript:
        # Imported lazily so the package imports cleanly even if the optional
        # dependency is missing.
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError as exc:  # pragma: no cover - trivial
            raise TranscriptError(
                "youtube-transcript-api is required for the YouTube source. "
                "Install it with: pip install youtube-transcript-api"
            ) from exc

        video_id = extract_video_id(url)
        api = YouTubeTranscriptApi()

        fetched, lang, generated = self._fetch_best(api, video_id, languages)
        segments = [Segment(s.text, float(s.start), float(s.duration)) for s in fetched]
        if not segments:
            raise TranscriptError(f"Transcript for {video_id} was empty.")

        title, author = _fetch_metadata(video_id)
        return Transcript(
            video_id=video_id,
            url=f"https://www.youtube.com/watch?v={video_id}",
            segments=segments,
            language=lang,
            is_generated=generated,
            title=title,
            author=author,
        )

    @staticmethod
    def _fetch_best(api, video_id: str, languages: list[str]):
        """Try requested languages (manual then generated), then fall back to
        any available transcript, translating it if possible."""
        try:
            listing = api.list(video_id)
        except Exception as exc:
            # Older/edge cases: fall straight to the flat fetch helper.
            try:
                fetched = api.fetch(video_id, languages=languages)
                return fetched, (languages[0] if languages else None), None
            except Exception:
                raise TranscriptError(
                    f"Could not retrieve a transcript for {video_id}: {exc}"
                ) from exc

        try:
            tr = listing.find_transcript(languages)
            return tr.fetch(), tr.language_code, tr.is_generated
        except Exception:
            pass

        # Fall back to the first available transcript, translating if we can.
        for tr in listing:
            try:
                if languages and tr.is_translatable:
                    target = languages[0]
                    if tr.language_code != target:
                        translated = tr.translate(target)
                        return translated.fetch(), target, tr.is_generated
                return tr.fetch(), tr.language_code, tr.is_generated
            except Exception:
                continue

        raise TranscriptError(f"No usable transcript found for {video_id}.")


register_source(YouTubeSource())
