"""Core data structures shared across the pipeline.

Everything here is a plain dataclass with no I/O or LLM dependencies so the
types stay trivially testable and serialisable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


def format_timestamp(seconds: float) -> str:
    """Render a start offset in seconds as ``mm:ss`` (or ``h:mm:ss``)."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


@dataclass
class Segment:
    """A single timed snippet of a transcript."""

    text: str
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class Transcript:
    """A fetched transcript plus whatever source metadata we could gather."""

    video_id: str
    url: str
    segments: list[Segment]
    language: str | None = None
    is_generated: bool | None = None
    title: str | None = None
    author: str | None = None

    @property
    def text(self) -> str:
        """The transcript as one whitespace-joined string."""
        return " ".join(s.text.replace("\n", " ").strip() for s in self.segments)

    @property
    def duration(self) -> float:
        return self.segments[-1].end if self.segments else 0.0

    @property
    def char_count(self) -> int:
        return len(self.text)

    def timestamped_text(self, block_seconds: int = 20) -> str:
        """Transcript grouped into ``block_seconds`` blocks, each prefixed
        with a ``[mm:ss]`` marker. Handy for humans and for giving the model
        something to cite."""
        from .analysis.chunking import to_blocks  # local import avoids cycle

        return "\n\n".join(
            f"[{format_timestamp(b.start)}] {b.text}" for b in to_blocks(self, block_seconds)
        )


@dataclass
class Component:
    """A single load-bearing element extracted from the video."""

    statement: str
    kind: str = "claim"  # claim | decision | tradeoff | comparison-verdict | recommendation | evidence | method
    why_load_bearing: str = ""
    evidence: str = ""
    timestamp: str | None = None
    confidence: str = "medium"  # high | medium | low


@dataclass
class Comparison:
    """A head-to-head verdict, when the video is comparative."""

    subject: str
    verdict: str
    basis: str = ""


@dataclass
class Analysis:
    """The structured output of the load-bearing analysis."""

    thesis: str = ""
    content_type: str = "other"
    components: list[Component] = field(default_factory=list)
    comparisons: list[Comparison] = field(default_factory=list)
    recommendation: str = ""
    depends_on: list[str] = field(default_factory=list)
    discarded_as_filler: list[str] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    raw: str | None = None  # populated only when structured parsing failed


@dataclass
class Report:
    """Everything the pipeline produces for one URL."""

    transcript: Transcript
    analysis: Analysis

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.transcript.video_id,
            "url": self.transcript.url,
            "title": self.transcript.title,
            "author": self.transcript.author,
            "language": self.transcript.language,
            "duration_seconds": round(self.transcript.duration, 1),
            "analysis": asdict(self.analysis),
        }
