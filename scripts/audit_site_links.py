#!/usr/bin/env python3
"""Audit internal links in the rendered Quarto site."""

from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
SITE_DIR = ROOT / "_site"
SITE_BASE = "math435-mathematical-imaging"
SKIP_SCHEMES = {"http", "https", "mailto", "tel", "data", "javascript"}


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self.anchors: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if value is None:
                continue
            if name in {"href", "src"}:
                self.links.append((name, value))
            if name in {"id", "name"}:
                self.anchors.add(value)


def parse_html(path: Path) -> LinkParser:
    parser = LinkParser()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser


def resolve_target(source: Path, link: str) -> tuple[Path, str] | None:
    parsed = urlparse(link)
    if parsed.scheme in SKIP_SCHEMES:
        return None
    if parsed.netloc:
        return None

    raw_path = unquote(parsed.path)
    fragment = unquote(parsed.fragment)

    if not raw_path:
        return source, fragment

    if raw_path.startswith("/"):
        parts = raw_path.lstrip("/").split("/")
        if parts and parts[0] == SITE_BASE:
            parts = parts[1:]
        target = SITE_DIR.joinpath(*parts)
    else:
        target = (source.parent / raw_path).resolve()

    if target.is_dir():
        target = target / "index.html"

    return target, fragment


def main() -> int:
    if not SITE_DIR.exists():
        print("Run ./scripts/quarto render before auditing links.", file=sys.stderr)
        return 2

    html_files = sorted(SITE_DIR.rglob("*.html"))
    parsed_files = {path: parse_html(path) for path in html_files}
    anchors = {path: parser.anchors for path, parser in parsed_files.items()}
    broken: list[str] = []

    for source, parser in parsed_files.items():
        for attr, link in parser.links:
            resolved = resolve_target(source, link)
            if resolved is None:
                continue
            target, fragment = resolved
            if not target.exists():
                broken.append(f"{source.relative_to(SITE_DIR)}: missing {attr} target {link}")
                continue
            if fragment and target.suffix == ".html" and fragment not in anchors.get(target, set()):
                broken.append(f"{source.relative_to(SITE_DIR)}: missing anchor #{fragment} in {target.relative_to(SITE_DIR)}")

    if broken:
        print("Broken internal links:")
        for item in broken:
            print(f"- {item}")
        return 1

    print(f"Checked {len(html_files)} HTML files; no broken internal links found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
