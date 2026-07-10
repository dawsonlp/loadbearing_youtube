# loadbearing

Extract a video transcript from a URL and expose its **load-bearing
components** — the claims, decisions, tradeoffs, and verdicts the video's
conclusion actually rests on. Not a generic "here's a summary": it isolates the
structural argument and sets aside the filler.

> Supersedes the older `youtube-summary` tool. Rebuilt as a linear Python
> pipeline with a pluggable, discoverable model layer.

## Why a pipeline, not an agent

The workflow is linear and fully known in advance:

```
URL → video-id → transcript (+metadata) → analyze → render
```

Only the *analyze* step is non-deterministic, so it's the only thing delegated
to an LLM. Everything else is plain, testable Python. An autonomous agent loop
would add cost, latency, and nondeterminism with no decision for it to make.
The pipeline exposes a clean `analyze(url) → Report` function, so wrapping it as
an MCP tool or Agent-SDK tool later is trivial (see *Extending*).

## Install

Requires Python 3.10+. The default provider is **Ollama** (local, no API key).

```bash
uv venv && uv pip install -e '.[dev]'
# optional cloud providers:
uv pip install -e '.[openai]'      # or [anthropic], or [all]
```

## Usage

```bash
# Just the transcript (no LLM)
loadbearing transcript "https://www.youtube.com/watch?v=8mY9wx_iMSU"
loadbearing transcript URL --no-timestamps -o transcript.txt

# Load-bearing analysis (uses the configured provider; Ollama by default)
loadbearing analyze "https://www.youtube.com/watch?v=8mY9wx_iMSU"
loadbearing analyze URL --provider ollama --model gemma4:e4b
loadbearing analyze URL --format json -o report.json

# Discover providers and the models each one can use
loadbearing providers
```

`lb` is a shorter alias for `loadbearing`. You can also run it as a module:
`python -m loadbearing analyze URL`.

### As a library

```python
from loadbearing import analyze, get_transcript

report = analyze("https://youtu.be/8mY9wx_iMSU", provider="ollama", model="gemma4:e4b")
print(report.analysis.thesis)
for c in report.analysis.components:
    print(c.timestamp, c.statement)

transcript = get_transcript("https://youtu.be/8mY9wx_iMSU")  # no LLM
print(transcript.text)
```

## Configuration

Set via environment (or a `.env`, if `python-dotenv` is installed). See
[.env.example](.env.example). CLI flags override env.

| Variable | Default | Meaning |
|---|---|---|
| `LOADBEARING_PROVIDER` | `ollama` | `ollama` \| `openai` \| `anthropic` |
| `LOADBEARING_MODEL` | provider default | Model for the selected provider |
| `LOADBEARING_LANGUAGES` | `en` | Transcript language preference |
| `LOADBEARING_MAX_CHARS` | `12000` | Single-pass threshold / chunk size |
| `LOADBEARING_MAX_TOKENS` | `2500` | Max output tokens for synthesis |
| `OLLAMA_HOST` / `OLLAMA_MODEL` | `localhost:11434` / `llama3.2` | Ollama settings |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | — / `gpt-4o-mini` | OpenAI settings |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` | — / `claude-3-5-haiku-latest` | Anthropic settings |

## How it works

1. **Source** (`sources/`) — resolve the URL to a transcript source (YouTube
   today) and fetch segments via `youtube-transcript-api`, plus title/author via
   oEmbed. Pluggable via `register_source`.
2. **Chunking** (`analysis/chunking.py`) — collapse segments into timestamped
   blocks, then pack blocks into character-bounded chunks with overlap.
   Timestamps survive as `[mm:ss]` markers.
3. **Analysis** (`analysis/loadbearing.py`) — short transcripts get one
   structured pass; long ones get **map-reduce**: extract candidate points per
   chunk, then synthesise/rank into the final load-bearing set.
4. **Render** (`render.py`) — Markdown or JSON.

The single "load-bearing" rubric lives in `analysis/prompts.py` and is shared by
both stages.

## Providers are pluggable and discoverable

Each provider declares `is_available()` (creds present / daemon reachable) and
`list_models()` (live discovery where possible — Ollama lists local models,
cloud SDKs list their catalog). `loadbearing providers` reports all three.

Add a backend by subclassing `LLMProvider` and registering it:

```python
from loadbearing.providers import register
from loadbearing.providers.base import LLMProvider, LLMResponse

@register
class MyProvider(LLMProvider):
    name = "myllm"
    default_model = "my-model"
    def complete(self, prompt, *, system=None, max_tokens=2000, temperature=0.2, json_mode=False):
        ...
        return LLMResponse(text=..., provider=self.name, model=self.model)
```

## Extending: expose as an agent tool

Because the core is one function, wrapping it for an agent is a few lines:

```python
# mcp_server.py (sketch)
from loadbearing import analyze
def analyze_video(url: str) -> dict:
    return analyze(url).to_dict()
```

Register `analyze_video` as an MCP tool or Agent-SDK `@tool` and any agent can
call the pipeline as a single, deterministic capability.

## Tests

```bash
uv run pytest        # deterministic parts: URL parsing, chunking, registry, analyzer (fake provider)
```

## License

MIT
