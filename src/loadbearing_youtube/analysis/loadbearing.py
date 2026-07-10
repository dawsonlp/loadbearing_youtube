"""The load-bearing analyzer: orchestrates the (optional) map stage and the
synthesis stage over an LLM provider."""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor

from ..models import Analysis, Comparison, Component, Transcript
from ..providers.base import LLMProvider
from . import prompts
from .chunking import chunk_transcript, fits_single_pass

logger = logging.getLogger(__name__)

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)

# Per-chunk candidate extraction never needs the full synthesis budget; cap it
# so we don't generate (and pay for) a huge output per chunk.
_MAP_MAX_TOKENS = 4000


def _parse_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model response, tolerating code
    fences, surrounding prose, and truncated output."""
    try:
        return json.loads(text)
    except Exception:
        pass
    match = _JSON_OBJ.search(text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return _salvage_truncated_json(text)


def _salvage_truncated_json(text: str) -> dict | None:
    """Recover a usable object from JSON that was cut off (e.g. the model hit
    its token cap mid-array). Trims to the last complete element and closes any
    still-open brackets, dropping the incomplete tail rather than losing
    everything."""
    start = text.find("{")
    if start == -1:
        return None
    s = text[start:]

    cut = s.rfind("}")
    if cut == -1:
        return None
    prefix = s[: cut + 1]

    # Walk the prefix to find which brackets remain open, ignoring braces that
    # appear inside strings.
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in prefix:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()

    closers = "".join("}" if b == "{" else "]" for b in reversed(stack))
    candidate = prefix.rstrip().rstrip(",") + closers
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _to_components(items: list[dict]) -> list[Component]:
    out = []
    for it in items or []:
        if not isinstance(it, dict) or not it.get("statement"):
            continue
        out.append(
            Component(
                statement=str(it.get("statement", "")).strip(),
                kind=str(it.get("kind", "claim")).strip() or "claim",
                why_load_bearing=str(it.get("why_load_bearing", "")).strip(),
                evidence=str(it.get("evidence", "")).strip(),
                timestamp=(it.get("timestamp") or None),
                confidence=str(it.get("confidence", "medium")).strip() or "medium",
            )
        )
    return out


def _to_comparisons(items: list[dict]) -> list[Comparison]:
    out = []
    for it in items or []:
        if not isinstance(it, dict) or not it.get("subject"):
            continue
        out.append(
            Comparison(
                subject=str(it.get("subject", "")).strip(),
                verdict=str(it.get("verdict", "")).strip(),
                basis=str(it.get("basis", "")).strip(),
            )
        )
    return out


class LoadBearingAnalyzer:
    def __init__(
        self,
        provider: LLMProvider,
        *,
        max_chars: int = 12000,
        max_tokens: int = 8000,
        max_workers: int = 4,
        on_progress=None,
    ):
        self.provider = provider
        self.max_chars = max_chars
        self.max_tokens = max_tokens
        # Bounded concurrency for the map stage. Cap keeps us under cloud rate
        # limits and avoids thrashing a single local Ollama daemon.
        self.max_workers = max(1, max_workers)
        self._progress = on_progress or (lambda msg: None)

    def analyze(self, transcript: Transcript) -> Analysis:
        if fits_single_pass(transcript, self.max_chars):
            self._progress("Analysing transcript in a single pass...")
            analysis = self._single_pass(transcript)
        else:
            analysis = self._map_reduce(transcript)
        analysis.provider = self.provider.name
        analysis.model = self.provider.model
        return analysis

    # --- single pass -----------------------------------------------------
    def _single_pass(self, transcript: Transcript) -> Analysis:
        prompt = prompts.SINGLE_PASS_INSTRUCTION + transcript.timestamped_text()
        resp = self.provider.complete(
            prompt,
            system=prompts.SYNTH_SYSTEM,
            max_tokens=self.max_tokens,
            json_mode=True,
        )
        return self._build_analysis(resp.text)

    # --- map / reduce ----------------------------------------------------
    def _map_reduce(self, transcript: Transcript) -> Analysis:
        chunks = chunk_transcript(transcript, max_chars=self.max_chars)
        workers = min(self.max_workers, len(chunks))
        self._progress(
            f"Transcript is long; mapping over {len(chunks)} chunks "
            f"({workers} in parallel)..."
        )

        # Map stage: chunks are independent, so run them concurrently. A failed
        # chunk degrades to no candidates rather than killing the whole run.
        # Results are reassembled in chunk order for stable synthesis input.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            per_chunk = list(pool.map(self._map_one_chunk, chunks))

        candidates: list[dict] = []
        for chunk_candidates in per_chunk:
            candidates.extend(chunk_candidates)

        self._progress(f"Synthesising {len(candidates)} candidate points...")
        candidate_json = json.dumps({"candidates": candidates}, ensure_ascii=False, indent=2)
        resp = self.provider.complete(
            prompts.SYNTH_INSTRUCTION + candidate_json,
            system=prompts.SYNTH_SYSTEM,
            max_tokens=self.max_tokens,
            json_mode=True,
        )
        return self._build_analysis(resp.text)

    def _map_one_chunk(self, chunk) -> list[dict]:
        """Extract candidate points from a single chunk. Runs on a worker
        thread; must not raise — a failed chunk contributes no candidates."""
        try:
            resp = self.provider.complete(
                prompts.MAP_INSTRUCTION + chunk.text,
                system=prompts.MAP_SYSTEM,
                max_tokens=_MAP_MAX_TOKENS,
                json_mode=True,
            )
            parsed = _parse_json(resp.text) or {}
            return parsed.get("candidates", []) or []
        except Exception as exc:
            logger.warning("map stage failed for chunk %s: %s", getattr(chunk, "index", "?"), exc)
            self._progress(f"  chunk {getattr(chunk, 'index', '?')} failed: {exc}")
            return []

    # --- shared ----------------------------------------------------------
    def _build_analysis(self, text: str) -> Analysis:
        data = _parse_json(text)
        if data is None:
            # Preserve the model output rather than lose it.
            return Analysis(
                thesis="(structured parsing failed — see raw output)",
                raw=text,
            )
        return Analysis(
            thesis=str(data.get("thesis", "")).strip(),
            content_type=str(data.get("content_type", "other")).strip() or "other",
            components=_to_components(data.get("components", [])),
            comparisons=_to_comparisons(data.get("comparisons", [])),
            recommendation=str(data.get("recommendation", "")).strip(),
            depends_on=[str(x).strip() for x in data.get("depends_on", []) or []],
            discarded_as_filler=[str(x).strip() for x in data.get("discarded_as_filler", []) or []],
        )
