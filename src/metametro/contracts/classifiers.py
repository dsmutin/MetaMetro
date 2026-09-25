"""Colour a CFA from Kraken2 or Kaiju contig calls.

These colours are classifier evidence. They are not simulation taxon ids.
A half-community database is the full parent pin's ``db`` genomes; the
metagenome and graph may be the half set.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Mapping

from metametro.contracts.colouring import paint_namespace
from metametro.errors import ContractError
from metametro.formats.cfa.model import CfaGraph


def parse_kraken2_output(path: Path) -> dict[str, list[int]]:
    """Map sequence id to taxids with positive k-mer mass.

    A line without the k-mer column uses the classified taxid. Taxid 0 is
    dropped. Unclassified rows contribute no colour.
    """
    found: dict[str, list[int]] = {}
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if text.strip() == "":
        raise ContractError([f"empty Kraken2 output: {path}"])
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            raise ContractError([f"kraken2 line has fewer than 3 columns: {line[:80]}"])
        seq_id = parts[1]
        taxa: dict[int, int] = {}
        if len(parts) >= 5:
            for token in parts[4].split():
                if ":" not in token:
                    continue
                tax_token, count_token = token.split(":", 1)
                if not tax_token.lstrip("-").isdigit() or not count_token.lstrip("-").isdigit():
                    continue
                tax_id = int(tax_token)
                count = int(count_token)
                if tax_id != 0 and count > 0:
                    taxa[tax_id] = taxa.get(tax_id, 0) + count
        if not taxa and parts[0] == "C":
            tax_id = int(parts[2])
            if tax_id != 0:
                taxa[tax_id] = 1
        found[seq_id] = sorted(taxa)
    return found


def parse_kaiju_output(path: Path) -> dict[str, list[int]]:
    """Map sequence id to the classified Kaiju taxid. Unclassified rows are empty."""
    found: dict[str, list[int]] = {}
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if text.strip() == "":
        raise ContractError([f"empty Kaiju output: {path}"])
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            raise ContractError([f"kaiju line has fewer than 3 columns: {line[:80]}"])
        seq_id = parts[1]
        if parts[0] == "C":
            tax_id = int(parts[2])
            found[seq_id] = [] if tax_id == 0 else [tax_id]
        else:
            found[seq_id] = []
    return found


def colour_from_calls(
    cfa: CfaGraph,
    calls: Mapping[str, list[int]],
    *,
    namespace: str,
    operation: str = "merge",
) -> CfaGraph:
    """Paint taxid colours from a contig-id map. Missing ids stay uncoloured."""
    node_values = {row["node_id"]: [str(tax_id) for tax_id in calls.get(row["node_id"], [])] for row in cfa.nodes}
    by_node = {node_id: set(values) for node_id, values in node_values.items()}
    edge_values: dict[str, list[str]] = {}
    for row in cfa.edges:
        shared = sorted(by_node.get(row["source"], set()) & by_node.get(row["target"], set()))
        edge_values[row["edge_id"]] = shared
    return paint_namespace(cfa, node_values, edge_values, namespace=namespace, operation=operation)


def write_node_fasta(cfa: CfaGraph, path: Path) -> Path:
    """Write CFA node sequences as FASTA. Headers are ``node_id``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for row in cfa.nodes:
        node_id = row["node_id"]
        sequence = cfa.sequences[node_id]
        lines.append(f">{node_id}")
        for start in range(0, len(sequence), 80):
            lines.append(sequence[start : start + 80])
    if not lines:
        raise ContractError(["classifier colouring needs at least one node"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def colour_from_kraken2_file(cfa: CfaGraph, path: Path, *, operation: str = "merge") -> CfaGraph:
    """Paint ``kraken2`` colours from an existing Kraken2 contig output."""
    return colour_from_calls(cfa, parse_kraken2_output(path), namespace="kraken2", operation=operation)


def colour_from_kaiju_file(cfa: CfaGraph, path: Path, *, operation: str = "merge") -> CfaGraph:
    """Paint ``kaiju`` colours from an existing Kaiju contig output."""
    return colour_from_calls(cfa, parse_kaiju_output(path), namespace="kaiju", operation=operation)


def run_kraken2(fasta: Path, db: Path, output: Path, *, threads: int = 4) -> Path:
    """Classify ``fasta`` with Kraken2. ``kraken2`` must be on PATH."""
    program = shutil.which("kraken2")
    if program is None:
        raise ContractError(["kraken2 is not on PATH"])
    if not (Path(db) / "hash.k2d").is_file():
        raise ContractError([f"Kraken2 database is missing hash.k2d: {db}"])
    output.parent.mkdir(parents=True, exist_ok=True)
    report = output.with_suffix(".report")
    completed = subprocess.run(
        [
            program,
            "--db",
            str(db),
            "--threads",
            str(threads),
            "--report",
            str(report),
            "--output",
            str(output),
            str(fasta),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ContractError([f"kraken2 exited {completed.returncode}: {(completed.stderr or '').strip()}"])
    if not output.is_file() or output.stat().st_size == 0:
        raise ContractError([f"kraken2 wrote no output: {output}"])
    return output


def run_kaiju(fasta: Path, db: Path, output: Path, *, threads: int = 4) -> Path:
    """Classify ``fasta`` with Kaiju. ``kaiju`` must be on PATH."""
    program = shutil.which("kaiju")
    if program is None:
        raise ContractError(["kaiju is not on PATH"])
    fmi = next(Path(db).glob("*.fmi"), None)
    nodes = Path(db) / "nodes.dmp"
    if fmi is None:
        raise ContractError([f"Kaiju FM-index is missing under {db}"])
    if not nodes.is_file():
        taxonomy = Path(db) / "taxonomy" / "nodes.dmp"
        if taxonomy.is_file():
            nodes = taxonomy
        else:
            raise ContractError([f"Kaiju nodes.dmp is missing under {db}"])
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            program,
            "-t",
            str(nodes),
            "-f",
            str(fmi),
            "-i",
            str(fasta),
            "-z",
            str(threads),
            "-o",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ContractError([f"kaiju exited {completed.returncode}: {(completed.stderr or '').strip()}"])
    if not output.is_file() or output.stat().st_size == 0:
        raise ContractError([f"kaiju wrote no output: {output}"])
    return output
