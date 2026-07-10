"""loadbearing — extract a video transcript and expose its load-bearing components.

Public API:

    from loadbearing import analyze, get_transcript, Report
    report = analyze("https://www.youtube.com/watch?v=...")
    print(report.analysis.thesis)
"""

from __future__ import annotations

__version__ = "0.1.4"

from .models import Analysis, Comparison, Component, Report, Segment, Transcript
from .pipeline import analyze, get_transcript

__all__ = [
    "__version__",
    "analyze",
    "get_transcript",
    "Report",
    "Transcript",
    "Analysis",
    "Component",
    "Comparison",
    "Segment",
]
