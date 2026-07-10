"""The linear pipeline that ties everything together.

    url --> source.fetch --> transcript --> analyzer.analyze --> Report

Deterministic orchestration in Python; the single non-deterministic step (the
analysis) is delegated to a pluggable provider.
"""

from __future__ import annotations

from typing import Callable

from .analysis import LoadBearingAnalyzer
from .config import Settings
from .models import Report, Transcript
from .providers import get_provider
from .sources import fetch_transcript

Progress = Callable[[str], None]


def get_transcript(
    url: str,
    languages: list[str] | None = None,
) -> Transcript:
    """Fetch just the transcript (no LLM involved)."""
    return fetch_transcript(url, languages)


def analyze(
    url: str,
    *,
    settings: Settings | None = None,
    languages: list[str] | None = None,
    provider: str | None = None,
    model: str | None = None,
    on_progress: Progress | None = None,
) -> Report:
    """Full pipeline: fetch transcript, then expose its load-bearing components."""
    settings = settings or Settings.from_env()
    progress = on_progress or (lambda msg: None)
    langs = languages or settings.languages

    progress(f"Fetching transcript for {url} ...")
    transcript = fetch_transcript(url, langs)
    progress(
        f"Transcript: {len(transcript.segments)} segments, "
        f"{transcript.char_count} chars"
        + (f" — “{transcript.title}”" if transcript.title else "")
    )

    llm = get_provider(provider or settings.provider, model or settings.model)
    progress(f"Provider: {llm.name} / {llm.model}")

    analyzer = LoadBearingAnalyzer(
        llm,
        max_chars=settings.max_chars,
        max_tokens=settings.max_tokens,
        on_progress=progress,
    )
    analysis = analyzer.analyze(transcript)
    return Report(transcript=transcript, analysis=analysis)
