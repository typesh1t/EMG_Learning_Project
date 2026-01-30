#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patch notebooks so they can import the project's `code.weekXX_*` modules when run
under IPython/Jupyter.

Problem:
  In Jupyter, the stdlib module `code` is often imported early (by IPython).
  Later, notebook cells like `from code.week06_preprocessing.filters import ...`
  fail with:
      No module named 'code.week06_preprocessing'; 'code' is not a package

Fix:
  After adding the project root to sys.path, add the project `code/` directory to
  the already-imported `code` module's `__path__`, effectively letting it behave
  like a package for our submodules.
"""

from __future__ import annotations

from pathlib import Path

import nbformat


SNIPPET_LINES = [
    "",
    "# --- Jupyter/IPython 兼容性修复 ---",
    "# IPython 往往会先导入标准库的 `code` 模块，导致 `code.weekXX_*` 不是包。",
    "# 这里把项目的 `code/` 目录挂到 `code.__path__`，让子模块可被正常导入。",
    "import code as _code",
    "if not hasattr(_code, '__path__'):",
    "    _code.__path__ = [str(project_root / 'code')]",
]


def patch_notebook(path: Path) -> bool:
    nb = nbformat.read(path, as_version=4)

    modified = False
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue

        source: str = cell.get("source") or ""
        if "_code.__path__" in source:
            continue

        # Find the first sys.path insertion that adds project_root.
        lines = source.splitlines()
        for idx, line in enumerate(lines):
            if "sys.path.insert(0, str(project_root))" not in line:
                continue

            # Insert the snippet right after this line, preserving existing comments.
            lines = lines[: idx + 1] + SNIPPET_LINES + lines[idx + 1 :]
            cell["source"] = "\n".join(lines) + ("\n" if source.endswith("\n") else "")
            modified = True
            break

        if modified:
            break

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
        print("Patched notebooks:")
        for p in changed:
            print(f"  - {p.relative_to(repo_root)}")
    else:
        print("No changes needed (notebooks already patched).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

