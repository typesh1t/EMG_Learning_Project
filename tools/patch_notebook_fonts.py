#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patch notebooks' matplotlib font settings to avoid "Glyph XXXX missing" warnings.

In some notebooks we use dingbat circled digits like "❶❷❸" in titles.
Common Chinese fonts (e.g., SimHei) may not include these glyphs, causing
matplotlib warnings when rendering figures in Jupyter.

Fix: extend `plt.rcParams['font.sans-serif']` with a symbol-capable font
(Windows: 'Segoe UI Symbol') plus a general fallback ('DejaVu Sans').
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat


FONT_ASSIGN_RE = re.compile(
    r"^(?P<prefix>\s*plt\.rcParams\['font\.sans-serif'\]\s*=\s*)\[(?P<body>.*)\](?P<suffix>\s*(#.*)?)$"
)


def _split_font_list(body: str) -> list[str]:
    # Very small parser for a python list literal content containing only string literals.
    # Keeps existing order; strips whitespace.
    items = []
    for part in body.split(","):
        item = part.strip()
        if not item:
            continue
        # Keep original quoting but normalize to plain string for comparisons.
        if (item.startswith("'") and item.endswith("'")) or (item.startswith('"') and item.endswith('"')):
            items.append(item[1:-1])
        else:
            items.append(item)
    return items


def _render_font_list(fonts: list[str]) -> str:
    return "[" + ", ".join(f"'{f}'" for f in fonts) + "]"


def patch_notebook(path: Path) -> bool:
    nb = nbformat.read(path, as_version=4)
    modified = False

    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        source: str = cell.get("source") or ""

        lines = source.splitlines()
        new_lines: list[str] = []
        changed_in_cell = False

        for line in lines:
            m = FONT_ASSIGN_RE.match(line)
            if not m:
                new_lines.append(line)
                continue

            fonts = _split_font_list(m.group("body"))

            # Add symbol-capable fallbacks (keep Chinese fonts first).
            for fallback in ["Segoe UI Symbol", "Arial Unicode MS", "DejaVu Sans"]:
                if fallback not in fonts:
                    fonts.append(fallback)

            new_line = m.group("prefix") + _render_font_list(fonts) + (m.group("suffix") or "")
            new_lines.append(new_line)
            changed_in_cell = True

        if changed_in_cell:
            cell["source"] = "\n".join(new_lines) + ("\n" if source.endswith("\n") else "")
            modified = True

    if modified:
        nbformat.write(nb, path)
    return modified


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    notebooks_dir = repo_root / "notebooks"
    if not notebooks_dir.exists():
        raise FileNotFoundError(f"notebooks dir not found: {notebooks_dir}")

    notebook_paths = sorted(notebooks_dir.glob("*.ipynb"))
    if not notebook_paths:
        print(f"No notebooks found under: {notebooks_dir}")
        return 0

    changed: list[Path] = []
    for nb_path in notebook_paths:
        if patch_notebook(nb_path):
            changed.append(nb_path)

    if changed:
        print("Patched notebooks (font fallbacks):")
        for p in changed:
            print(f"  - {p.relative_to(repo_root)}")
    else:
        print("No changes needed (font fallbacks already present).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

