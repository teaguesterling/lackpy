"""Format converters: execution log ↔ .ipynb ↔ markdown.

Converts between the driver's CellExecutionEvent log and two output formats:
  to_notebook() — .ipynb with lackpy metadata for round-trip fidelity.
                  Loading it back via from_notebook() + kernel = "Restart and Run All."
  render_markdown() — the literate document the model should have written
                      after recovery fixes. Skipped/aborted cells are omitted.
"""

from __future__ import annotations

from typing import Any

from ..parser import Cell, Frontmatter
from .driver import CellExecutionEvent

_CODE_CELL_TYPES = {"code", "hidden", "gather", "scratch", "continue", "read", "write", "diff"}


def to_notebook(log: list[CellExecutionEvent], frontmatter: Frontmatter) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for event in log:
        cell = event.cell
        nb_cell: dict[str, Any] = {
            "metadata": {
                "lackpy": {
                    "cell_type": cell.cell_type,
                    "annotation_args": cell.annotation_args,
                    "status": event.status,
                    "recovery_attempts": event.recovery_attempts,
                    "generation": event.generation,
                }
            },
            "outputs": [],
        }

        if cell.cell_type == "prose":
            nb_cell["cell_type"] = "markdown"
            nb_cell["source"] = cell.content
        else:
            nb_cell["cell_type"] = "code"
            nb_cell["source"] = cell.content
            nb_cell["execution_count"] = event.cell_index + 1
            output_text = event.result.output if event.result and event.result.output else ""
            if output_text:
                nb_cell["outputs"].append({
                    "output_type": "stream",
                    "name": "stdout",
                    "text": output_text,
                })

        cells.append(nb_cell)

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "lackpy": {
                "frontmatter": {
                    "echo": frontmatter.echo,
                    "output": frontmatter.output,
                    "interpreter": frontmatter.interpreter,
                },
            },
            "kernelspec": {
                "display_name": "Lackpy Literate",
                "language": "python",
                "name": "lackpy-literate",
            },
        },
        "cells": cells,
    }


def from_notebook(nb: dict[str, Any]) -> tuple[Frontmatter, list[Cell]]:
    fm_data = nb.get("metadata", {}).get("lackpy", {}).get("frontmatter", {})
    frontmatter = Frontmatter(
        echo=fm_data.get("echo", "true"),
        output=fm_data.get("output", "auto"),
        interpreter=fm_data.get("interpreter", "python"),
    )

    cells: list[Cell] = []
    for nb_cell in nb.get("cells", []):
        lackpy_meta = nb_cell.get("metadata", {}).get("lackpy", {})
        cell_type = lackpy_meta.get("cell_type", "code")
        annotation_args = lackpy_meta.get("annotation_args", {})

        if cell_type == "prose" or nb_cell.get("cell_type") == "markdown":
            cell_type = "prose"

        cells.append(Cell(
            cell_type=cell_type,
            content=nb_cell.get("source", ""),
            annotation_args=annotation_args,
        ))

    return frontmatter, cells


def render_markdown(log: list[CellExecutionEvent], frontmatter: Frontmatter) -> str:
    parts: list[str] = []

    has_non_default_fm = (
        frontmatter.echo != "true"
        or frontmatter.output != "auto"
        or frontmatter.interpreter != "python"
    )
    if has_non_default_fm:
        parts.append("---")
        if frontmatter.echo != "true":
            parts.append(f"echo: {frontmatter.echo}")
        if frontmatter.output != "auto":
            parts.append(f"output: {frontmatter.output}")
        if frontmatter.interpreter != "python":
            parts.append(f"interpreter: {frontmatter.interpreter}")
        parts.append("---")
        parts.append("")

    for event in log:
        if event.status in ("skipped", "aborted", "pending"):
            continue

        cell = event.cell
        if cell.cell_type == "prose":
            parts.append(cell.content)
            parts.append("")
        elif cell.cell_type == "code":
            parts.append("```lackpy")
            parts.append(cell.content)
            parts.append("```")
            parts.append("")
        elif cell.cell_type in _CODE_CELL_TYPES:
            annotation = f"@{cell.cell_type}"
            if cell.annotation_args.get("path"):
                annotation += f"({cell.annotation_args['path']})"
            parts.append(f"```lackpy {annotation}")
            parts.append(cell.content)
            parts.append("```")
            parts.append("")

    return "\n".join(parts).rstrip("\n") + "\n"
