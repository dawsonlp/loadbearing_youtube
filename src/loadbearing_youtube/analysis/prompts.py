"""Prompt templates for the load-bearing analysis.

The definition of "load-bearing" is stated once here and reused so the map and
synthesis stages share the same rubric.
"""

from __future__ import annotations

LOAD_BEARING_DEFINITION = """\
A LOAD-BEARING component is a claim, decision, comparison verdict, tradeoff, \
method, or recommendation that the video's conclusion actually rests on: if you \
removed or negated it, the takeaway would change. It is the structural argument, \
not the decoration. NON-load-bearing content includes intros, restated points, \
anecdotes, hype, sponsor reads, and calls to action — note these only briefly as \
filler."""

MAP_SYSTEM = f"""You extract the structural argument from a transcript chunk.

{LOAD_BEARING_DEFINITION}

You will receive one chunk of a longer transcript with inline [mm:ss] timestamps. \
List only the candidate load-bearing statements found in THIS chunk. Be precise \
and quote-grounded; do not invent content that is not present."""

MAP_INSTRUCTION = """\
From the chunk below, extract candidate load-bearing components. Return JSON:
{
  "candidates": [
    {
      "statement": "one crisp sentence capturing the point",
      "kind": "claim|decision|tradeoff|comparison-verdict|recommendation|evidence|method",
      "evidence": "the reason or example the speaker gives, if any",
      "timestamp": "mm:ss from the nearest marker, or null"
    }
  ]
}
If the chunk is only filler, return {"candidates": []}.

CHUNK:
"""

SYNTH_SYSTEM = f"""You are a rigorous analyst distilling a video into its \
structural argument.

{LOAD_BEARING_DEFINITION}

Consolidate the provided candidate points into the final set of load-bearing \
components: merge duplicates, drop filler, and keep each component distinct and \
essential. Preserve timestamps where available. Do not add claims that are not \
supported by the candidates."""

SYNTH_INSTRUCTION = """\
Produce the final analysis as a single JSON object with this shape:
{
  "thesis": "one-sentence statement of the video's central claim",
  "content_type": "comparison|tutorial|explainer|review|interview|news|other",
  "components": [
    {
      "statement": "crisp sentence",
      "kind": "claim|decision|tradeoff|comparison-verdict|recommendation|evidence|method",
      "why_load_bearing": "what would change if this were removed or false",
      "evidence": "supporting reason/example from the video",
      "timestamp": "mm:ss or null",
      "confidence": "high|medium|low"
    }
  ],
  "comparisons": [
    {"subject": "what is being compared", "verdict": "the outcome", "basis": "why"}
  ],
  "recommendation": "the video's bottom-line recommendation, if any",
  "depends_on": ["short labels of the components the recommendation rests on"],
  "discarded_as_filler": ["brief notes on notable non-load-bearing content"]
}
Order components from most to least load-bearing. Use [] for sections that do not apply.

CANDIDATE POINTS:
"""

# For short transcripts we skip the map stage and do one structured pass.
SINGLE_PASS_INSTRUCTION = """\
Analyse the transcript below and produce the final analysis as a single JSON \
object with this shape:
{
  "thesis": "one-sentence statement of the video's central claim",
  "content_type": "comparison|tutorial|explainer|review|interview|news|other",
  "components": [
    {
      "statement": "crisp sentence",
      "kind": "claim|decision|tradeoff|comparison-verdict|recommendation|evidence|method",
      "why_load_bearing": "what would change if this were removed or false",
      "evidence": "supporting reason/example from the video",
      "timestamp": "mm:ss or null",
      "confidence": "high|medium|low"
    }
  ],
  "comparisons": [{"subject": "...", "verdict": "...", "basis": "..."}],
  "recommendation": "bottom-line recommendation, if any",
  "depends_on": ["labels of components the recommendation rests on"],
  "discarded_as_filler": ["brief notes on non-load-bearing content"]
}
Order components from most to least load-bearing. Use [] where a section does not apply.

TRANSCRIPT:
"""
