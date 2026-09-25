"""Mandatory checks for stacked colourings, composition, decaying, and classifiers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from metametro.bench import load_bench_cdbg, namespaces_for
from metametro.bench.build import build
from metametro.bench.colourings import listed_colourings, registered_names
from metametro.bench.data.universal.catalog import classifier_pin
from metametro.bench.registry import resolve
from metametro.contracts.classifiers import colour_from_calls, parse_kaiju_output, parse_kraken2_output
from metametro.contracts.colour_filter import filter_colours
from metametro.contracts.colouring import colour_by_reads, paint_namespace
from metametro.contracts.composition import colour_by_composition, kmeans_onehot
from metametro.contracts.decaying import colour_decaying, decaying_distributions
from metametro.errors import ContractError
from metametro.formats.cfa.io import load_cfa
from metametro.formats.cfa.model import CfaGraph

pytestmark = pytest.mark.mandatory


def _tiny() -> CfaGraph:
    return CfaGraph(
        metadata={"schema_version": "1.0", "graph_id": "tiny", "graph_type": "de_bruijn", "k": 3},
        sequences={"a": "ACG", "b": "CGT"},
        nodes=[{"node_id": "a"}, {"node_id": "b"}],
        edges=[{"edge_id": "e", "source": "a", "target": "b"}],
        node_header=["node_id"],
        edge_header=["edge_id", "source", "target"],
    )


def test_registry_lists_migrated_colourings() -> None:
    names = registered_names()
    assert names[0] == "as_built"
    assert names[-1] == "decaying"
    listed = {name: (namespace, auto) for name, namespace, auto in listed_colourings()}
    assert listed["composition_kmeans"] == ("composition", True)
    assert listed["kraken2"] == ("kraken2", True)
    assert listed["kaiju"] == ("kaiju", True)
    assert listed["read_accession"] == ("accession", False)


def test_paint_namespace_does_not_reuse_ids() -> None:
    graph = paint_namespace(_tiny(), {"a": ["taxon_1"], "b": ["taxon_2"]}, {"e": ["taxon_1"]}, namespace="taxon")
    merged = colour_by_reads(graph, [("r1", "s0", "ACG")], ["s0"], min_edge_kmer_density=1, operation="merge")
    namespaces = {row["namespace"] for row in merged.colors or []}
    assert namespaces == {"taxon", "sample"}
    ids = [int(row["color_id"]) for row in merged.colors or []]
    assert len(ids) == len(set(ids))


def test_composition_clusters_are_not_genome_ids() -> None:
    coloured = colour_by_composition(_tiny(), n_clusters=2, seed=0, operation="replace")
    assert {row["namespace"] for row in coloured.colors or []} == {"composition"}
    values = {row["value"] for row in coloured.colors or []}
    assert values <= {"cluster_0", "cluster_1"}
    onehot = kmeans_onehot(np.array([[0.0, 1.0], [1.0, 0.0], [0.1, 0.9]], dtype=np.float64), n_clusters=2, seed=0)
    assert onehot.dtype == np.uint8
    assert onehot.shape[1] == 2


def test_decaying_spreads_to_a_neighbour() -> None:
    graph = paint_namespace(_tiny(), {"a": ["red"], "b": []}, {"e": []}, namespace="mark", operation="replace")
    leaked = decaying_distributions({"a": {0: 1.0}, "b": {}}, [{"source": "a", "target": "b", "weight": 1.0}], decay=0.5, iterations=1)
    assert leaked["a"][0] == pytest.approx(1.0)
    coloured = colour_decaying(graph, decay=0.5, iterations=2, threshold=0.2)
    decaying = {row["value"] for row in coloured.colors or [] if row["namespace"] == "decaying"}
    assert "red" in decaying


def test_kraken_and_kaiju_parsers_drop_unclassified(tmp_path: Path) -> None:
    kraken = tmp_path / "k.tsv"
    kraken.write_text("C\ta\t562\t3\t562:10 0:1\nU\tb\t0\t3\t0:3\n", encoding="utf-8")
    kaiju = tmp_path / "j.tsv"
    kaiju.write_text("C\ta\t562\nU\tb\t0\n", encoding="utf-8")
    assert parse_kraken2_output(kraken)["a"] == [562]
    assert parse_kraken2_output(kraken)["b"] == []
    assert parse_kaiju_output(kaiju) == {"a": [562], "b": []}
    coloured = colour_from_calls(_tiny(), {"a": [562], "b": []}, namespace="kraken2", operation="replace")
    assert any(row["value"] == "562" and row["namespace"] == "kraken2" for row in coloured.colors or [])


def test_empty_classifier_output_is_an_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty.tsv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ContractError, match="empty Kraken2"):
        parse_kraken2_output(empty)


def test_half_classifier_pin_is_the_parent() -> None:
    half = resolve("4domain_family_100_half")
    parent = resolve("4domain_family_100")
    assert classifier_pin(half) == classifier_pin(parent)
    assert half.parent == parent.name


def test_inprocess_build_stacks_composition_and_decaying(tmp_path: Path) -> None:
    result = build("bubble_strain_2", outdir=tmp_path / "out")
    assert "as_built" in result.colourings
    assert "composition_kmeans" in result.colourings
    assert "decaying" in result.colourings
    assert "read_accession" not in result.colourings
    cfa = load_cfa(result.outdir / "cfa")
    namespaces = {row["namespace"] for row in cfa.colors or []}
    assert "taxon" in namespaces
    assert "composition" in namespaces
    assert "decaying" in namespaces
    selected = filter_colours(cfa, namespaces=("taxon", "composition"))
    kept = {row["namespace"] for row in selected.colors or []}
    assert kept == {"taxon", "composition"}
    cdbg = load_bench_cdbg(result.outdir, namespaces=namespaces_for(("composition_kmeans",)))
    assert all(int(row["color_id"]) in {int(item["color_id"]) for item in cdbg.colors or []} for row in cdbg.colors or [])
    assert {row["namespace"] for row in cdbg.colors or []} == {"composition"}


def test_read_accession_is_explicit(tmp_path: Path) -> None:
    result = build("bubble_reads_2", outdir=tmp_path / "reads", colourings=("as_built", "read_depth"))
    cfa = load_cfa(result.outdir / "cfa")
    namespaces = {row["namespace"] for row in cfa.colors or []}
    assert "sample" in namespaces
    assert "accession" not in namespaces
    assert "strain_A" not in (result.outdir / "cfa" / "colors.tsv").read_text(encoding="utf-8")
