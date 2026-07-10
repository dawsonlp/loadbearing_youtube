"""Analyzer tests using a fake provider — no network, fully deterministic."""

import json

from loadbearing_youtube.analysis.loadbearing import LoadBearingAnalyzer
from loadbearing_youtube.models import Segment, Transcript
from loadbearing_youtube.providers.base import LLMProvider, LLMResponse
from loadbearing_youtube.render import render_markdown


class FakeProvider(LLMProvider):
    name = "fake"
    default_model = "fake-1"

    def __init__(self, payload, model=None):
        super().__init__(model)
        self._payload = payload

    def complete(self, prompt, *, system=None, max_tokens=2000, temperature=0.2, json_mode=False):
        return LLMResponse(text=self._payload, provider=self.name, model=self.model)


ANALYSIS_JSON = json.dumps(
    {
        "thesis": "Model choice is now a workflow-design decision.",
        "content_type": "comparison",
        "components": [
            {
                "statement": "Cost and rate limits decide the daily driver.",
                "kind": "claim",
                "why_load_bearing": "It drives the final recommendation.",
                "evidence": "GPT is ~50% cheaper.",
                "timestamp": "01:07",
                "confidence": "high",
            }
        ],
        "comparisons": [{"subject": "browser use", "verdict": "GPT wins", "basis": "reliability"}],
        "recommendation": "Use GPT as daily driver.",
        "depends_on": ["cost"],
        "discarded_as_filler": ["intro"],
    }
)


def _transcript():
    segs = [Segment(text=f"point {i}", start=i * 2.0, duration=2.0) for i in range(10)]
    return Transcript(video_id="x", url="u", segments=segs, title="Test", author="A")


def test_single_pass_parses_structured_output():
    analyzer = LoadBearingAnalyzer(FakeProvider(ANALYSIS_JSON), max_chars=100000)
    analysis = analyzer.analyze(_transcript())
    assert analysis.thesis.startswith("Model choice")
    assert analysis.content_type == "comparison"
    assert len(analysis.components) == 1
    assert analysis.components[0].timestamp == "01:07"
    assert analysis.provider == "fake"
    assert analysis.recommendation


def test_parsing_survives_prose_wrapped_json():
    wrapped = "Sure! Here is the analysis:\n```json\n" + ANALYSIS_JSON + "\n```\nHope that helps."
    analyzer = LoadBearingAnalyzer(FakeProvider(wrapped), max_chars=100000)
    analysis = analyzer.analyze(_transcript())
    assert len(analysis.components) == 1


def test_unparseable_output_preserved_as_raw():
    analyzer = LoadBearingAnalyzer(FakeProvider("total gibberish, no json"), max_chars=100000)
    analysis = analyzer.analyze(_transcript())
    assert analysis.raw is not None


def test_markdown_render_contains_sections():
    from loadbearing_youtube.models import Report

    analyzer = LoadBearingAnalyzer(FakeProvider(ANALYSIS_JSON), max_chars=100000)
    report = Report(transcript=_transcript(), analysis=analyzer.analyze(_transcript()))
    md = render_markdown(report)
    assert "Load-bearing components" in md
    assert "Recommendation" in md
    assert "Thesis" in md
