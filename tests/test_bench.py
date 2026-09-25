"""Mandatory: benchbuild writes three graphs and a stable identity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from metametro.bench import build
from metametro.bench.colourings import BuildContext, register, unregister
from metametro.bench.leakage import assert_no_target_leak
from metametro.bench.registry import list_specs
from metametro.bench.scoring.assembly.bubble_strain_2 import expected_action
from metametro.bench.scoring.assembly.universal.lengths import n50
from metametro.bench.scoring.assembly_binning.universal.contig_f1 import contig_f1
from metametro.bench.scoring.profiling.universal.abundance import l1, presence_f1
from metametro.cli import main
from metametro.contracts.colouring import colour_cfa
from metametro.errors import ContractError
from metametro.formats.cfa.io import load_cfa
from metametro.formats.cdbg.io import load_cdbg
from metametro.formats.cgt.io import load_cgt

pytestmark = pytest.mark.mandatory


def test_registry_lists_inprocess_benches() -> None:
    """The in-process benchmarks are registered under their new names."""
    names = {spec.name for spec in list_specs()}
    assert "bubble_strain_2" in names
    assert "bubble_reads_2" in names
    assert "synthetic_reads_2" in names
    assert "mock_bubble" in names
    assert "strong100" not in names


def test_benchbuild_strain_is_stable_and_coloured(tmp_path: Path) -> None:
    """Two builds of the two-taxon bubble share an identity and write three graphs."""
    first = build("bubble_strain_2", outdir=tmp_path / "a")
    second = build("strain_bubble", outdir=tmp_path / "b")
    assert first.identity == second.identity
    assert first.colourings == ("as_built",)
    for folder in (first.outdir, second.outdir):
        assert (folder / "cfa" / "metadata.yaml").is_file()
        assert (folder / "cdbg" / "metadata.yaml").is_file()
        assert (folder / "cgt" / "metadata.yaml").is_file()
        assert expected_action(folder) == "retain"
    cfa = load_cfa(first.outdir / "cfa")
    assert {row["node_id"] for row in cfa.nodes} == {"S", "A", "B", "T"}
    assert cfa.sequences["A"] == "ATATCG"
    assert_no_target_leak(cfa)
    text = (first.outdir / "cfa" / "nodes.tsv").read_text(encoding="utf-8")
    assert "expected_action" not in text
    cdbg = load_cdbg(first.outdir / "cdbg")
    cgt = load_cgt(first.outdir / "cgt")
    assert cgt.num_nodes == len(cdbg.unitigs)


def test_new_colouring_is_applied(tmp_path: Path) -> None:
    """A colouring registered after the benchmark still runs."""

    def mark(graph, _ctx: BuildContext):
        return colour_cfa(
            graph,
            node_colors={row["node_id"]: [7] for row in graph.nodes},
            operation="merge",
            colors=list(graph.colors or []) + [{"color_id": "7", "namespace": "mark", "value": "unit"}],
        )

    register("unit_test_mark", mark)
    try:
        result = build("bubble_error_1", outdir=tmp_path / "out")
    finally:
        unregister("unit_test_mark")
    assert "unit_test_mark" in result.colourings
    cfa = load_cfa(result.outdir / "cfa")
    assert any(row["namespace"] == "mark" for row in cfa.colors or [])


def test_target_column_is_rejected() -> None:
    """A truth column on the CFA is not a legal benchmark graph."""
    from metametro.bench.data.universal.bubbles import strain_bubble

    graph = strain_bubble()
    graph.metadata["features"]["node"]["truth_taxon_id"] = "int"
    graph.node_header = [*graph.node_header, "truth_taxon_id"]
    with pytest.raises(ContractError, match="target leaked"):
        assert_no_target_leak(graph)


def test_reads_benchmark_keeps_genome_ids_out_of_the_graph(tmp_path: Path) -> None:
    """Read colouring uses samples. Genome ids stay in ground truth."""
    result = build("bubble_reads_2", outdir=tmp_path / "reads")
    assert "read_depth" in result.colourings
    cfa = load_cfa(result.outdir / "cfa")
    namespaces = {row["namespace"] for row in cfa.colors or []}
    assert "sample" in namespaces
    assert "truth" not in namespaces
    truth = (result.outdir / "ground_truth" / "read_to_genome.tsv").read_text(encoding="utf-8")
    assert "strain_A" in truth
    assert "strain_A" not in (result.outdir / "cfa" / "colors.tsv").read_text(encoding="utf-8")


def test_synthetic_fifty_is_deterministic(tmp_path: Path) -> None:
    """The 50-bubble generator does not depend on the output path."""
    left = build("bubble_strain_3_n50", outdir=tmp_path / "left")
    right = build("bubble_strain_3_n50", outdir=tmp_path / "right")
    assert left.identity == right.identity
    assert left.n_nodes if hasattr(left, "n_nodes") else load_cfa(left.outdir / "cfa")
    cfa = load_cfa(left.outdir / "cfa")
    assert len(cfa.nodes) > 50
    truth = (left.outdir / "ground_truth" / "bubbles.tsv").read_text(encoding="utf-8").strip().splitlines()
    assert len(truth) == 51
    assert sum(line.endswith("\tpop") for line in truth) == 25


def test_cli_benchbuild_list(capsys: pytest.CaptureFixture[str]) -> None:
    """``metametro benchbuild --list`` prints the two-taxon bubble."""
    assert main(["benchbuild", "--list"]) == 0
    assert "bubble_strain_2" in capsys.readouterr().out


def test_default_outdir_shape(tmp_path: Path) -> None:
    """The default tree is data/bench/{name}/{assembler}/{properties}."""
    result = build("mock_bubble", root=tmp_path)
    assert result.outdir == tmp_path / "data" / "bench" / "mock_bubble" / "inprocess" / "k3"
    assert (result.outdir / "cgt" / "indptr.npy").is_file()


def test_scoring_helpers() -> None:
    """Universal scorers match closed-form cases."""
    assert n50([10, 20, 30]) == 30
    assert l1({1: 0.5, 2: 0.5}, {1: 1.0, 2: 1.0}) == 0.0
    assert presence_f1({1: 1.0}, {1: 1.0, 2: 1.0}) == pytest.approx(2 / 3)
    perfect = contig_f1(np.array([0, 0, 1, 1]), np.array([5, 5, 9, 9]))
    assert perfect == pytest.approx(1.0)


def test_strain_matches_bubbleblower_when_present() -> None:
    """Node sequences and taxon colours match BubbleBlower's strain bubble."""
    sibling = Path(__file__).resolve().parents[2] / "BubbleBlower" / "src"
    if not (sibling / "bubbleblower" / "fixtures.py").is_file():
        pytest.skip("BubbleBlower checkout is not beside metametro")
    import sys

    sys.path.insert(0, str(sibling))
    from bubbleblower.fixtures import strain_bubble

    graph = strain_bubble()
    sequences = sorted((unitig.sequence, tuple(unitig.color_ids)) for unitig in graph.cdbg.unitigs)
    from metametro.bench.data.universal.bubbles import strain_bubble as our_strain
    from metametro.converters.cfa_to_cdbg import cfa_to_cdbg

    ours = cfa_to_cdbg(our_strain())
    our_sequences = sorted((unitig.sequence, tuple(unitig.color_ids)) for unitig in ours.unitigs)
    assert our_sequences == sequences
