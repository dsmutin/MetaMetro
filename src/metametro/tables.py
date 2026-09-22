"""TSV and YAML helpers used by the on-disk graph formats."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a tab-separated table. The first row is the header."""
    text = path.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip() != ""]
    if not lines:
        raise ValueError(f"empty table: {path}")
    header = lines[0].split("\t")
    if any(column == "" for column in header):
        raise ValueError(f"empty header column in {path}")
    if len(header) != len(set(header)):
        raise ValueError(f"duplicate header column in {path}")
    rows: list[dict[str, str]] = []
    for line_number, line in enumerate(lines[1:], start=2):
        cells = line.split("\t")
        if len(cells) != len(header):
            raise ValueError(
                f"{path}:{line_number}: expected {len(header)} columns, found {len(cells)}"
            )
        rows.append(dict(zip(header, cells)))
    return header, rows


def write_tsv(path: Path, header: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a tab-separated table. Values are stringified without extra spaces."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(header)]
    for row in rows:
        lines.append("\t".join("" if row.get(column) is None else str(row[column]) for column in header))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def write_yaml(path: Path, data: Mapping[str, Any]) -> None:
    """Write a YAML mapping with stable key order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(dict(data), sort_keys=False, allow_unicode=False)
    path.write_text(text, encoding="utf-8")
