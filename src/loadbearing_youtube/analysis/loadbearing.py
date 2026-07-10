"""The load-bearing analyzer: orchestrates the (optional) map stage and the
synthesis stage over an LLM provider."""

from __future__ import annotations

import json
import re

from ..models import Analysis, Comparison, Component, Transcript
from ..providers.base import LLMProvider
from . import prompts
from .chunking import chunk_transcript, fits_single_pass

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model response, tolerating code
    fences and surrounding prose."""
    try:
        return json.loads(text)
    except Exception:
        pass
    match = _JSON_OBJ.search(text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
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
        max_tokens: int = 2500,
        on_progress=None,
    ):
        self.provider = provider
        self.max_chars = max_chars
        self.max_tokens = max_tokens
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
        self._progress(f"Transcript is long; mapping over {len(chunks)} chunks...")
        candidates: list[dict] = []
        for chunk in chunks:
            self._progress(f"  chunk {chunk.index + 1}/{len(chunks)} ([{chunk.start:.0f}s])")
            resp = self.provider.complete(
                prompts.MAP_INSTRUCTION + chunk.text,
                system=prompts.MAP_SYSTEM,
                max_tokens=1500,
                json_mode=True,
            )
            parsed = _parse_json(resp.text) or {}
            candidates.extend(parsed.get("candidates", []) or [])

        self._progress(f"Synthesising {len(candidates)} candidate points...")
        candidate_json = json.dumps({"candidates": candidates}, ensure_ascii=False, indent=2)
        resp = self.provider.complete(
            prompts.SYNTH_INSTRUCTION + candidate_json,
            system=prompts.SYNTH_SYSTEM,
            max_tokens=self.max_tokens,
            json_mode=True,
        )
        return self._build_analysis(resp.text)

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
