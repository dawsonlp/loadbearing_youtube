"""Deterministic, timestamp-aware chunking.

The transcript is first collapsed into fixed-duration *blocks* (each carrying a
start offset), then blocks are packed into *chunks* that fit a character
budget. Timestamps survive into the chunk text as ``[mm:ss]`` markers so the
model can cite them.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import Transcript, format_timestamp

# Rough chars-per-token; only used to translate a token budget into a char one.
CHARS_PER_TOKEN = 4


@dataclass
class Block:
    start: float
    text: str


@dataclass
class Chunk:
    index: int
    start: float
    end: float
    text: str  # includes inline [mm:ss] markers


def to_blocks(transcript: Transcript, block_seconds: int = 20) -> list[Block]:
    """Group segments into ~``block_seconds`` blocks."""
    blocks: list[Block] = []
    buf: list[str] = []
    block_start: float | None = None
    for seg in transcript.segments:
        if block_start is None:
            block_start = seg.start
        buf.append(seg.text.replace("\n", " ").strip())
        if seg.start - block_start >= block_seconds:
            blocks.append(Block(block_start, " ".join(buf).strip()))
            buf, block_start = [], None
    if buf and block_start is not None:
        blocks.append(Block(block_start, " ".join(buf).strip()))
    return blocks


def chunk_transcript(
    transcript: Transcript,
    max_chars: int = 12000,
    overlap_blocks: int = 1,
    block_seconds: int = 20,
) -> list[Chunk]:
    """Pack timestamped blocks into character-bounded chunks with a small
    block overlap for continuity."""
    blocks = to_blocks(transcript, block_seconds)
    if not blocks:
        return []

    rendered = [f"[{format_timestamp(b.start)}] {b.text}" for b in blocks]

    chunks: list[Chunk] = []
    i = 0
    n = len(blocks)
    while i < n:
        cur: list[str] = []
        size = 0
        j = i
        while j < n:
            piece = rendered[j]
            if cur and size + len(piece) + 1 > max_chars:
                break
            cur.append(piece)
            size += len(piece) + 1
            j += 1
        chunks.append(
            Chunk(
                index=len(chunks),
                start=blocks[i].start,
                end=blocks[j - 1].start,
                text="\n".join(cur),
            )
        )
        if j >= n:
            break
        i = max(j - overlap_blocks, i + 1)
    return chunks


def fits_single_pass(transcript: Transcript, max_chars: int = 12000) -> bool:
    return transcript.char_count <= max_chars
