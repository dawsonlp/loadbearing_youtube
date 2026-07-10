"""Rendering of transcripts and analyses to text/markdown/json."""

from __future__ import annotations

import json

from .models import Report, Transcript, format_timestamp


def render_transcript(transcript: Transcript, timestamps: bool = True, block_seconds: int = 20) -> str:
    if timestamps:
        return transcript.timestamped_text(block_seconds)
    return transcript.text


def render_json(report: Report) -> str:
    return json.dumps(report.to_dict(), ensure_ascii=False, indent=2)


def render_markdown(report: Report) -> str:
    t = report.transcript
    a = report.analysis
    lines: list[str] = []

    heading = t.title or f"YouTube video {t.video_id}"
    lines.append(f"# Load-bearing analysis — {heading}")
    meta = []
    if t.author:
        meta.append(t.author)
    meta.append(t.url)
    meta.append(f"{format_timestamp(t.duration)} runtime")
    if a.provider:
        meta.append(f"analysed by {a.provider}/{a.model}")
    lines.append("*" + " · ".join(meta) + "*")
    lines.append("")

    if a.raw:
        lines.append("> Structured parsing failed; raw model output follows.\n")
        lines.append(a.raw)
        return "\n".join(lines)

    if a.thesis:
        lines.append(f"**Thesis.** {a.thesis}")
        lines.append("")

    if a.components:
        lines.append("## Load-bearing components")
        lines.append("")
        for i, c in enumerate(a.components, 1):
            ts = f" `[{c.timestamp}]`" if c.timestamp else ""
            lines.append(f"**{i}. {c.statement}**{ts}  ")
            tags = f"_{c.kind} · confidence: {c.confidence}_"
            lines.append(tags + "  ")
            if c.why_load_bearing:
                lines.append(f"- Why it's load-bearing: {c.why_load_bearing}")
            if c.evidence:
                lines.append(f"- Evidence: {c.evidence}")
            lines.append("")

    if a.comparisons:
        lines.append("## Comparisons")
        lines.append("")
        lines.append("| Subject | Verdict | Basis |")
        lines.append("|---|---|---|")
        for cmp in a.comparisons:
            basis = cmp.basis.replace("|", "\\|")
            lines.append(f"| {cmp.subject} | {cmp.verdict} | {basis} |")
        lines.append("")

    if a.recommendation:
        lines.append("## Recommendation")
        lines.append("")
        lines.append(a.recommendation)
        lines.append("")
        if a.depends_on:
            lines.append("*Rests on:* " + ", ".join(a.depends_on))
            lines.append("")

    if a.discarded_as_filler:
        lines.append("## Set aside as filler")
        lines.append("")
        for item in a.discarded_as_filler:
            lines.append(f"- {item}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
