"""Command-line interface.

Uses only the standard library (argparse) so the core install stays light.

    loadbearing transcript URL [--timestamps] [-o FILE]
    loadbearing analyze    URL [--provider P] [--model M] [--format md|json] [-o FILE]
    loadbearing providers
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .config import Settings
from .providers import discover
from .render import render_json, render_markdown, render_transcript
from .sources.base import TranscriptError


def _progress(msg: str) -> None:
    print(msg, file=sys.stderr)


def _write(text: str, output: str | None) -> None:
    if output:
        with open(output, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"Wrote {output}", file=sys.stderr)
    else:
        print(text)


def cmd_transcript(args: argparse.Namespace) -> int:
    from .pipeline import get_transcript

    langs = _parse_langs(args.languages)
    transcript = get_transcript(args.url, langs)
    _write(render_transcript(transcript, timestamps=args.timestamps), args.output)
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from .pipeline import analyze

    settings = Settings.from_env()
    report = analyze(
        args.url,
        settings=settings,
        languages=_parse_langs(args.languages),
        provider=args.provider,
        model=args.model,
        on_progress=_progress,
    )
    text = render_json(report) if args.format == "json" else render_markdown(report)
    _write(text, args.output)
    return 0


def cmd_providers(_args: argparse.Namespace) -> int:
    for p in discover():
        status = "available" if p["available"] else "not configured"
        print(f"{p['name']:12} [{status}]  default: {p['default_model']}")
        for m in p["models"][:25]:
            print(f"    - {m}")
        if len(p["models"]) > 25:
            print(f"    … and {len(p['models']) - 25} more")
    return 0


def _parse_langs(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loadbearing",
        description="Extract a video transcript and expose its load-bearing components.",
    )
    parser.add_argument("--version", action="version", version=f"loadbearing {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("transcript", help="Fetch a transcript (no LLM).")
    t.add_argument("url")
    t.add_argument("--languages", "-l", help="Comma-separated language codes (e.g. en,es).")
    ts = t.add_mutually_exclusive_group()
    ts.add_argument("--timestamps", dest="timestamps", action="store_true", default=True)
    ts.add_argument("--no-timestamps", dest="timestamps", action="store_false")
    t.add_argument("--output", "-o", help="Write to file instead of stdout.")
    t.set_defaults(func=cmd_transcript)

    a = sub.add_parser("analyze", help="Fetch and analyse load-bearing components.")
    a.add_argument("url")
    a.add_argument("--provider", "-p", help="LLM provider (default: env or ollama).")
    a.add_argument("--model", "-m", help="Model name for the provider.")
    a.add_argument("--languages", "-l", help="Comma-separated language codes.")
    a.add_argument("--format", "-f", choices=["md", "json"], default="md")
    a.add_argument("--output", "-o", help="Write to file instead of stdout.")
    a.set_defaults(func=cmd_analyze)

    p = sub.add_parser("providers", help="List providers, availability, and models.")
    p.set_defaults(func=cmd_providers)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except TranscriptError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # keep CLI failures tidy
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
