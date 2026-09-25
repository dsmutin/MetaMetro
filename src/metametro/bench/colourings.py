"""Colourings plugged into MetaMetro.

``benchbuild`` applies every auto colouring registered here that can run on
the build. A benchmark does not freeze that list, so a colouring added after
the benchmark was written is applied on the next build. Colourings with
``auto=False`` run only when ``--colouring`` names them.

Namespaces:

- ``as_built`` keeps colours already on the graph (taxon, transit, roxel)
- ``read_depth`` → ``sample``
- ``composition_kmeans`` → ``composition``
- ``kraken2`` → ``kraken2``
- ``kaiju`` → ``kaiju``
- ``decaying`` → ``decaying``
- ``read_accession`` → ``accession`` (explicit only; simulated accessions leak truth)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from metametro.contracts.classifiers import (
    colour_from_kaiju_file,
    colour_from_kraken2_file,
    run_kaiju,
    run_kraken2,
    write_node_fasta,
)
from metametro.contracts.colouring import colour_by_read_accessions, colour_by_reads
from metametro.contracts.composition import colour_by_composition
from metametro.contracts.decaying import colour_decaying
from metametro.formats.cfa.model import CfaGraph


@dataclass
class BuildContext:
    """Inputs a colouring may use. Missing inputs make that colouring unavailable."""

    reads: list[tuple[str, str, str]] | None = None
    samples: list[str] | None = None
    min_vertex_depth: int = 1
    min_edge_kmer_density: int = 2
    selected: tuple[str, ...] | None = None
    work_dir: Path | None = None
    kraken_db: Path | None = None
    kaiju_db: Path | None = None
    kraken_output: Path | None = None
    kaiju_output: Path | None = None
    composition_k: int = 4
    composition_n_clusters: int = 8
    composition_seed: int = 0
    decay: float = 0.5
    decay_iterations: int = 4
    decay_threshold: float = 0.5


@dataclass(frozen=True)
class Colouring:
    """One registered colouring."""

    name: str
    apply: Callable[[CfaGraph, BuildContext], CfaGraph]
    available: Callable[[CfaGraph, BuildContext], bool]
    auto: bool = True
    namespace: str | None = None


_REGISTRY: dict[str, Colouring] = {}

NAMESPACES = {
    "as_built": None,
    "read_depth": "sample",
    "composition_kmeans": "composition",
    "kraken2": "kraken2",
    "kaiju": "kaiju",
    "decaying": "decaying",
    "read_accession": "accession",
}


def register(
    name: str,
    apply: Callable[[CfaGraph, BuildContext], CfaGraph],
    *,
    available: Callable[[CfaGraph, BuildContext], bool] | None = None,
    auto: bool = True,
    namespace: str | None = None,
) -> None:
    """Register ``name``. Registering the same name again replaces the colouring."""
    if not name or any(char.isspace() for char in name):
        raise ValueError(f"colouring name must be a non-empty token, got {name!r}")
    ready = available if available is not None else (lambda _graph, _ctx: True)
    _REGISTRY[name] = Colouring(name=name, apply=apply, available=ready, auto=auto, namespace=namespace)


def unregister(name: str) -> None:
    """Remove a colouring. Used by tests that install a temporary colouring."""
    _REGISTRY.pop(name, None)


def registered_names() -> tuple[str, ...]:
    """Return colouring names in registration order."""
    return tuple(_REGISTRY)


def listed_colourings() -> tuple[tuple[str, str | None, bool], ...]:
    """Return ``(name, namespace, auto)`` for every registered colouring."""
    return tuple((item.name, item.namespace, item.auto) for item in _REGISTRY.values())


def colouring_namespace(name: str) -> str | None:
    """Return the colour dictionary namespace written by ``name``, if it is one namespace."""
    if name in _REGISTRY:
        return _REGISTRY[name].namespace
    return NAMESPACES.get(name)


def applicable(graph: CfaGraph, ctx: BuildContext) -> tuple[Colouring, ...]:
    """Return colourings that can run, honouring an explicit selection."""
    if ctx.selected is None:
        chosen = {name: item for name, item in _REGISTRY.items() if item.auto}
        missing: list[str] = []
    else:
        chosen = {name: _REGISTRY[name] for name in ctx.selected if name in _REGISTRY}
        missing = [name for name in ctx.selected if name not in _REGISTRY]
    if missing:
        from metametro.errors import ContractError

        raise ContractError([f"unknown colouring: {', '.join(missing)}"])
    return tuple(item for item in chosen.values() if item.available(graph, ctx))


def _as_built(graph: CfaGraph, _ctx: BuildContext) -> CfaGraph:
    return graph


def _has_k(graph: CfaGraph) -> bool:
    k = graph.metadata.get("k")
    return isinstance(k, int) and not isinstance(k, bool) and k > 0


def _already_coloured(graph: CfaGraph) -> bool:
    for row in list(graph.nodes) + list(graph.edges):
        if str(row.get("color_set", "")).strip():
            return True
    return False


def _operation(graph: CfaGraph) -> str:
    return "merge" if _already_coloured(graph) else "replace"


def _reads_available(graph: CfaGraph, ctx: BuildContext) -> bool:
    return bool(ctx.reads) and bool(ctx.samples) and _has_k(graph)


def _read_depth(graph: CfaGraph, ctx: BuildContext) -> CfaGraph:
    assert ctx.reads is not None and ctx.samples is not None
    return colour_by_reads(
        graph,
        ctx.reads,
        ctx.samples,
        min_vertex_depth=ctx.min_vertex_depth,
        min_edge_kmer_density=ctx.min_edge_kmer_density,
        operation=_operation(graph),
        namespace="sample",
    )


def _composition_available(graph: CfaGraph, _ctx: BuildContext) -> bool:
    return len(graph.nodes) >= 2 and all(graph.sequences.get(row["node_id"], "") for row in graph.nodes)


def _composition(graph: CfaGraph, ctx: BuildContext) -> CfaGraph:
    return colour_by_composition(
        graph,
        k=ctx.composition_k,
        n_clusters=ctx.composition_n_clusters,
        seed=ctx.composition_seed,
        operation=_operation(graph),
    )


def _decaying_available(graph: CfaGraph, _ctx: BuildContext) -> bool:
    return bool(graph.edges) and _already_coloured(graph)


def _decaying(graph: CfaGraph, ctx: BuildContext) -> CfaGraph:
    return colour_decaying(
        graph,
        decay=ctx.decay,
        iterations=ctx.decay_iterations,
        threshold=ctx.decay_threshold,
        operation=_operation(graph),
    )


def _kraken_available(graph: CfaGraph, ctx: BuildContext) -> bool:
    if not graph.nodes:
        return False
    if ctx.kraken_output is not None and Path(ctx.kraken_output).is_file():
        return True
    db = ctx.kraken_db
    return db is not None and (Path(db) / "hash.k2d").is_file() and shutil_which("kraken2") is not None


def _kaiju_available(graph: CfaGraph, ctx: BuildContext) -> bool:
    if not graph.nodes:
        return False
    if ctx.kaiju_output is not None and Path(ctx.kaiju_output).is_file():
        return True
    db = ctx.kaiju_db
    if db is None:
        return False
    has_fmi = next(Path(db).glob("*.fmi"), None) is not None
    return has_fmi and shutil_which("kaiju") is not None


def shutil_which(name: str) -> str | None:
    """Wrap ``shutil.which`` so tests can patch this module."""
    import shutil

    return shutil.which(name)


def _work(ctx: BuildContext) -> Path:
    if ctx.work_dir is not None:
        return Path(ctx.work_dir)
    return Path("work")


def _kraken2(graph: CfaGraph, ctx: BuildContext) -> CfaGraph:
    if ctx.kraken_output is not None and Path(ctx.kraken_output).is_file():
        return colour_from_kraken2_file(graph, Path(ctx.kraken_output), operation=_operation(graph))
    assert ctx.kraken_db is not None
    fasta = write_node_fasta(graph, _work(ctx) / "classifier" / "nodes.fna")
    output = run_kraken2(fasta, Path(ctx.kraken_db), _work(ctx) / "classifier" / "contigs.kraken")
    return colour_from_kraken2_file(graph, output, operation=_operation(graph))


def _kaiju(graph: CfaGraph, ctx: BuildContext) -> CfaGraph:
    if ctx.kaiju_output is not None and Path(ctx.kaiju_output).is_file():
        return colour_from_kaiju_file(graph, Path(ctx.kaiju_output), operation=_operation(graph))
    assert ctx.kaiju_db is not None
    fasta = write_node_fasta(graph, _work(ctx) / "classifier" / "nodes.fna")
    output = run_kaiju(fasta, Path(ctx.kaiju_db), _work(ctx) / "classifier" / "contigs.kaiju")
    return colour_from_kaiju_file(graph, output, operation=_operation(graph))


def _accession(graph: CfaGraph, ctx: BuildContext) -> CfaGraph:
    assert ctx.reads is not None
    return colour_by_read_accessions(
        graph,
        ctx.reads,
        min_vertex_depth=ctx.min_vertex_depth,
        min_edge_kmer_density=ctx.min_edge_kmer_density,
        operation=_operation(graph),
    )


def _install_builtins() -> None:
    if "as_built" not in _REGISTRY:
        register("as_built", _as_built, namespace=None)
    if "read_depth" not in _REGISTRY:
        register("read_depth", _read_depth, available=_reads_available, namespace="sample")
    if "composition_kmeans" not in _REGISTRY:
        register("composition_kmeans", _composition, available=_composition_available, namespace="composition")
    if "kraken2" not in _REGISTRY:
        register("kraken2", _kraken2, available=_kraken_available, namespace="kraken2")
    if "kaiju" not in _REGISTRY:
        register("kaiju", _kaiju, available=_kaiju_available, namespace="kaiju")
    if "read_accession" not in _REGISTRY:
        register(
            "read_accession",
            _accession,
            available=_reads_available,
            auto=False,
            namespace="accession",
        )
    if "decaying" not in _REGISTRY:
        register("decaying", _decaying, available=_decaying_available, namespace="decaying")


_install_builtins()
