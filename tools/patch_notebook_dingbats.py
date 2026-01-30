#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replace dingbat negative circled digits (❶❷❸...) with normal circled digits (①②③...).

Matplotlib often cannot render dingbat digits with common CJK fonts (SimHei/微软雅黑),
which leads to warnings like:
  UserWarning: Glyph 10102 (DINGBAT NEGATIVE CIRCLED DIGIT ONE) missing from current font.

Using ①②③ keeps the teaching style while avoiding missing glyph warnings.
"""

from __future__ import annotations

from pathlib import Path

import nbformat


REPLACEMENTS = str.maketrans({
    "❶": "①",
    "❷": "②",
    "❸": "③",
    "❹": "④",
    "❺": "⑤",
    "❻": "⑥",
    "❼": "⑦",
    "❽": "⑧",
    "❾": "⑨",
    "❿": "⑩",
})


def patch_notebook(path: Path) -> bool:
    nb = nbformat.read(path, as_version=4)
    modified = False

    for cell in nb.cells:
        source = cell.get("source")
        if not source:
            continue
        new_source = source.translate(REPLACEMENTS)
        if new_source != source:
            cell["source"] = new_source
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
        print("Patched notebooks (dingbat -> circled digits):")
        for p in changed:
            print(f"  - {p.relative_to(repo_root)}")
    else:
        print("No dingbat digits found; no changes needed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

