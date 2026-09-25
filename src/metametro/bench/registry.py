"""Catalog of benchmarks. Strong100 and other prebuilt bundles are not listed."""

from __future__ import annotations

from metametro.bench.data.universal.catalog import community_specs, external_specs
from metametro.bench.data.universal.inprocess import (
    prepare_bubble_error_1,
    prepare_bubble_nested_2,
    prepare_bubble_reads_2,
    prepare_bubble_shared_2,
    prepare_bubble_strain_2,
    prepare_bubble_strain_3_n50,
    prepare_mock_bubble,
    prepare_mock_chain,
    prepare_synthetic_reads_2,
)
from metametro.bench.spec import BenchSpec
from metametro.formats.cfa.model import CfaGraph
from metametro.bench.colourings import BuildContext
from pathlib import Path
from typing import Callable

Prepare = Callable[[Path, BuildContext], CfaGraph]

SPECS: dict[str, BenchSpec] = {
    "bubble_strain_2": BenchSpec(
        name="bubble_strain_2",
        assembler="inprocess",
        properties="taxa2",
        summary="Two taxa, one bubble. BubbleBlower toy. Debubbler truth: retain.",
        scoring=("assembly",),
        kind="inprocess",
        legacy_names=("strain_bubble", "bubbleblower_toy"),
    ),
    "bubble_error_1": BenchSpec(
        name="bubble_error_1",
        assembler="inprocess",
        properties="taxon1",
        summary="One taxon, one error bubble. Debubbler truth: pop.",
        scoring=("assembly",),
        kind="inprocess",
        legacy_names=("error_bubble",),
    ),
    "bubble_nested_2": BenchSpec(
        name="bubble_nested_2",
        assembler="inprocess",
        properties="taxa2",
        summary="Two taxa, one outer bubble and one inner bubble.",
        scoring=("assembly",),
        kind="inprocess",
        legacy_names=("nested_bubbles",),
    ),
    "bubble_shared_2": BenchSpec(
        name="bubble_shared_2",
        assembler="inprocess",
        properties="taxa2",
        summary="Two taxa sharing one node. No bubble.",
        scoring=("assembly",),
        kind="inprocess",
        legacy_names=("shared_duplicate_node",),
    ),
    "bubble_strain_3_n50": BenchSpec(
        name="bubble_strain_3_n50",
        assembler="inprocess",
        properties="seed42",
        summary="Three strains and 50 disjoint bubbles (25 strain, 25 error), seed 42.",
        scoring=("assembly",),
        kind="inprocess",
        legacy_names=("bubble_benchmark",),
    ),
    "bubble_reads_2": BenchSpec(
        name="bubble_reads_2",
        assembler="inprocess",
        properties="k5_len12_seed1",
        summary="Two strains, read length 12, de Bruijn k=5, 20 reads per sample.",
        scoring=("assembly",),
        kind="inprocess",
        legacy_names=("bubbleblower_reads",),
    ),
    "mock_bubble": BenchSpec(
        name="mock_bubble",
        assembler="inprocess",
        properties="k3",
        summary="MetaMetro four-node bubble fixture.",
        scoring=("assembly",),
        kind="inprocess",
    ),
    "mock_chain": BenchSpec(
        name="mock_chain",
        assembler="inprocess",
        properties="k3",
        summary="MetaMetro six-node chain that compacts to three unitigs.",
        scoring=("assembly",),
        kind="inprocess",
    ),
    "synthetic_reads_2": BenchSpec(
        name="synthetic_reads_2",
        assembler="inprocess",
        properties="k8_len16_seed1",
        summary="Two synthetic genomes and an in-process de Bruijn graph at k=8.",
        scoring=("assembly", "profiling"),
        kind="inprocess",
    ),
}

SPECS.update(community_specs())
SPECS.update(external_specs())

PREPARE: dict[str, Prepare] = {
    "bubble_strain_2": prepare_bubble_strain_2,
    "bubble_error_1": prepare_bubble_error_1,
    "bubble_nested_2": prepare_bubble_nested_2,
    "bubble_shared_2": prepare_bubble_shared_2,
    "bubble_strain_3_n50": prepare_bubble_strain_3_n50,
    "bubble_reads_2": prepare_bubble_reads_2,
    "mock_bubble": prepare_mock_bubble,
    "mock_chain": prepare_mock_chain,
    "synthetic_reads_2": prepare_synthetic_reads_2,
}

_LEGACY = {
    alias: spec.name
    for spec in SPECS.values()
    for alias in spec.legacy_names
}


def resolve(name: str) -> BenchSpec:
    """Return the spec for a canonical name or a legacy alias."""
    from metametro.errors import ContractError

    canonical = _LEGACY.get(name, name)
    spec = SPECS.get(canonical)
    if spec is None:
        known = ", ".join(sorted(SPECS))
        raise ContractError([f"unknown benchmark {name!r}. Known benchmarks: {known}"])
    return spec


def list_specs() -> tuple[BenchSpec, ...]:
    """Return specs in a stable order."""
    return tuple(SPECS[name] for name in sorted(SPECS))
