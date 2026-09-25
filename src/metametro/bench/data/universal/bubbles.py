"""Hand-built bubble graphs migrated from BubbleBlower.

The evaluation action (retain or pop) is written only by the caller, under
``ground_truth/``, and is not a column of these graphs.
"""

from __future__ import annotations

import math
import random

from metametro.bench.data.universal.cfa_records import records_to_cfa
from metametro.formats.cfa.model import CfaGraph

_TAXA = [
    {"color_id": "0", "namespace": "taxon", "value": "taxon_1"},
    {"color_id": "1", "namespace": "taxon", "value": "taxon_2"},
    {"color_id": "2", "namespace": "taxon", "value": "taxon_3"},
]


def _colors(names: list[str]) -> list[dict[str, str]]:
    wanted = set(names)
    return [row for row in _TAXA if row["value"] in wanted]


def strain_bubble() -> CfaGraph:
    """Two taxa, one bubble. Ground truth for a debubbler is retain."""
    return records_to_cfa(
        graph_id="bubble_strain_2",
        colors=_colors(["taxon_1", "taxon_2"]),
        nodes=[
            {"id": "S", "sequence": "ACGTACGT", "colors": [0, 1], "coverage": 120.0},
            {"id": "A", "sequence": "ATATCG", "colors": [0], "coverage": 30.0},
            {"id": "B", "sequence": "CGCGTA", "colors": [1], "coverage": 90.0},
            {"id": "T", "sequence": "GGCCTTAA", "colors": [0, 1], "coverage": 120.0},
        ],
        links=[
            {"id": "eSA", "source": "S", "target": "A", "colors": [0], "coverage": 30.0},
            {"id": "eAT", "source": "A", "target": "T", "colors": [0], "coverage": 30.0},
            {"id": "eSB", "source": "S", "target": "B", "colors": [1], "coverage": 90.0},
            {"id": "eBT", "source": "B", "target": "T", "colors": [1], "coverage": 90.0},
        ],
    )


def error_bubble() -> CfaGraph:
    """One taxon, one bubble, with a low-coverage branch. Ground truth is pop."""
    return records_to_cfa(
        graph_id="bubble_error_1",
        colors=_colors(["taxon_2"]),
        nodes=[
            {"id": "S", "sequence": "ACGTACGT", "colors": [1], "coverage": 91.0},
            {"id": "A", "sequence": "ATATCGAT", "colors": [1], "coverage": 90.0},
            {"id": "E", "sequence": "CGCGTACG", "colors": [1], "coverage": 1.0},
            {"id": "T", "sequence": "GGCCTTAA", "colors": [1], "coverage": 91.0},
        ],
        links=[
            {"id": "eSA", "source": "S", "target": "A", "colors": [1], "coverage": 90.0},
            {"id": "eAT", "source": "A", "target": "T", "colors": [1], "coverage": 90.0},
            {"id": "eSE", "source": "S", "target": "E", "colors": [1], "coverage": 1.0},
            {"id": "eET", "source": "E", "target": "T", "colors": [1], "coverage": 1.0},
        ],
    )


def nested_bubbles() -> CfaGraph:
    """Two taxa. One outer bubble and one inner bubble on a side path."""
    return records_to_cfa(
        graph_id="bubble_nested_2",
        colors=_colors(["taxon_1", "taxon_2"]),
        nodes=[
            {"id": "S", "sequence": "AAAAAA", "colors": [0, 1], "coverage": 40.0},
            {"id": "A", "sequence": "CCCCCC", "colors": [0], "coverage": 20.0},
            {"id": "U", "sequence": "GGGGGG", "colors": [1], "coverage": 20.0},
            {"id": "P", "sequence": "TTTTTT", "colors": [1], "coverage": 10.0},
            {"id": "Q", "sequence": "ACACAC", "colors": [1], "coverage": 10.0},
            {"id": "V", "sequence": "GTGTGT", "colors": [1], "coverage": 20.0},
            {"id": "B", "sequence": "CACACA", "colors": [1], "coverage": 20.0},
            {"id": "T", "sequence": "ATATAT", "colors": [0, 1], "coverage": 40.0},
        ],
        links=[
            {"id": "eSA", "source": "S", "target": "A", "colors": [0], "coverage": 20.0},
            {"id": "eAT", "source": "A", "target": "T", "colors": [0], "coverage": 20.0},
            {"id": "eSB", "source": "S", "target": "B", "colors": [1], "coverage": 20.0},
            {"id": "eBT", "source": "B", "target": "T", "colors": [1], "coverage": 20.0},
            {"id": "eTU", "source": "T", "target": "U", "colors": [1], "coverage": 20.0},
            {"id": "eUP", "source": "U", "target": "P", "colors": [1], "coverage": 10.0},
            {"id": "eUQ", "source": "U", "target": "Q", "colors": [1], "coverage": 10.0},
            {"id": "ePV", "source": "P", "target": "V", "colors": [1], "coverage": 10.0},
            {"id": "eQV", "source": "Q", "target": "V", "colors": [1], "coverage": 10.0},
            {"id": "eVT", "source": "V", "target": "T", "colors": [1], "coverage": 20.0},
        ],
    )


def shared_node() -> CfaGraph:
    """Two taxa sharing one intermediate node. This graph has no bubble."""
    return records_to_cfa(
        graph_id="bubble_shared_2",
        colors=_colors(["taxon_1", "taxon_2"]),
        nodes=[
            {"id": "S", "sequence": "ACGTACGT", "colors": [0, 1], "coverage": 100.0},
            {"id": "X", "sequence": "ATGCCATG", "colors": [0, 1], "coverage": 100.0},
            {"id": "T", "sequence": "GGCCTTAA", "colors": [0, 1], "coverage": 100.0},
        ],
        links=[
            {"id": "eSX", "source": "S", "target": "X", "colors": [0, 1], "coverage": 100.0},
            {"id": "eXT", "source": "X", "target": "T", "colors": [0, 1], "coverage": 100.0},
        ],
    )


def _poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    limit = math.exp(-lam)
    count = 0
    product = 1.0
    while product > limit:
        count += 1
        product *= rng.random()
    return count - 1


def _tag(index: int) -> str:
    alphabet = "ACGT"
    digits: list[str] = []
    value = index + 1
    while value:
        digits.append(alphabet[value % 4])
        value //= 4
    return "".join(digits) or "A"


def _band(rng: random.Random, kind: str) -> float:
    spans = {
        "very_low": (0.2, 1.0),
        "medium": (1.0, 3.0),
        "borderline": (3.0, 7.0),
        "hard": (7.0, 12.0),
    }
    low, high = spans[kind]
    return rng.uniform(low, high)


def synthetic_bubbles(
    *,
    n_strains: int = 3,
    n_bubbles: int = 50,
    n_error_bubbles: int = 25,
    abundance: tuple[float, ...] = (60.0, 40.0, 20.0),
    error_rate: float = 0.01,
    seed: int = 42,
) -> tuple[CfaGraph, list[dict[str, str]]]:
    """Build disjoint bubbles. The truth table is not stored on the graph.

    The draw matches BubbleBlower's seed-42 generator: 25 strain bubbles and
    25 error bubbles across three strains.
    """
    if n_error_bubbles > n_bubbles:
        raise ValueError("n_error_bubbles cannot exceed n_bubbles")
    if len(abundance) < n_strains:
        raise ValueError("abundance must cover every strain")
    rng = random.Random(seed)
    n_bio = n_bubbles - n_error_bubbles
    plan: list[tuple[str, tuple]] = []
    pairs = [(0, 1), (1, 2), (0, 2)]
    for index in range(min(10, n_bio)):
        plan.append(("strain", pairs[index % len(pairs)]))
    for index in range(min(10, max(0, n_bio - 10))):
        owner = index % n_strains
        other = (owner + 1) % n_strains
        plan.append(("strain", (owner, other)))
    three = tuple(range(n_strains))
    while len(plan) < n_bio:
        plan.append(("strain", three))
    error_kinds = (["very_low"] * 10) + (["medium"] * 8) + (["borderline"] * 5) + (["hard"] * 2)
    for index in range(n_error_bubbles):
        kind = error_kinds[index] if index < len(error_kinds) else "medium"
        plan.append(("error", (index % n_strains, kind)))
    nodes: list[dict] = []
    links: list[dict] = []
    truth: list[dict[str, str]] = []
    for index, (kind, spec) in enumerate(plan):
        bubble_id = f"B{index + 1:03d}"
        source = f"{bubble_id}S"
        sink = f"{bubble_id}T"
        if kind == "strain":
            strains = tuple(int(item) for item in spec)
            coverages = [float(_poisson(rng, abundance[strain])) for strain in strains]
            coverages = [max(value, 1.0) for value in coverages]
            expected = "retain"
            strain_names = [f"strain_{chr(ord('A') + strain)}" for strain in strains]
        else:
            owner = int(spec[0])
            band = str(spec[1])
            true_cov = float(max(_poisson(rng, abundance[owner]), 1))
            error_cov = max(_band(rng, band), error_rate * true_cov)
            strains = (owner, owner)
            coverages = [true_cov, error_cov]
            expected = "pop"
            strain_names = [f"strain_{chr(ord('A') + owner)}"]
        source_cov = sum(coverages)
        tag = _tag(index)
        nodes.append(
            {
                "id": source,
                "sequence": f"ACGT{'AC' * (index % 5)}TT{tag}",
                "colors": sorted({int(item) for item in strains}),
                "coverage": source_cov,
            }
        )
        nodes.append(
            {
                "id": sink,
                "sequence": f"GGCC{'GG' * (index % 4)}AA{tag}",
                "colors": sorted({int(item) for item in strains}),
                "coverage": source_cov,
            }
        )
        branch_ids: list[str] = []
        for branch_index, (strain, coverage) in enumerate(zip(strains, coverages)):
            branch_id = f"{bubble_id}{chr(ord('A') + branch_index)}"
            branch_ids.append(branch_id)
            base = "ATGC" if kind == "strain" else "ACGT"[branch_index % 4] * 4
            sequence = (base * 4)[: 8 + branch_index] + f"{index:02d}"[-2:]
            sequence = "".join(ch if ch in "ACGT" else "A" for ch in sequence)
            if len(sequence) < 6:
                sequence = (sequence + "ACGTAC")[:6]
            nodes.append(
                {
                    "id": branch_id,
                    "sequence": sequence + tag + ("A" if branch_index == 0 else "C" * branch_index),
                    "colors": [int(strain)],
                    "coverage": coverage,
                }
            )
            links.append(
                {
                    "id": f"{bubble_id}s{branch_index}",
                    "source": source,
                    "target": branch_id,
                    "colors": [int(strain)],
                    "coverage": coverage,
                }
            )
            links.append(
                {
                    "id": f"{bubble_id}t{branch_index}",
                    "source": branch_id,
                    "target": sink,
                    "colors": [int(strain)],
                    "coverage": coverage,
                }
            )
        truth.append(
            {
                "bubble_id": bubble_id,
                "type": "error" if kind == "error" else "strain",
                "strain_ids": ",".join(strain_names),
                "source_id": source,
                "sink_id": sink,
                "branch_ids": ",".join(branch_ids),
                "expected_action": expected,
            }
        )
    colors = [
        {"color_id": str(index), "namespace": "taxon", "value": f"strain_{chr(ord('A') + index)}"}
        for index in range(n_strains)
    ]
    graph = records_to_cfa(
        graph_id="bubble_strain_3_n50",
        nodes=nodes,
        links=links,
        colors=colors,
    )
    return graph, truth
