"""Colourings plugged into MetaMetro.

``benchbuild`` applies every colouring registered here that can run on the
build. A benchmark does not freeze that list, so a colouring added after the
benchmark was written is applied on the next build.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from metametro.contracts.colouring import colour_by_reads
from metametro.formats.cfa.model import CfaGraph


@dataclass
class BuildContext:
    """Inputs a colouring may use. Missing inputs make that colouring unavailable."""

    reads: list[tuple[str, str, str]] | None = None
    samples: list[str] | None = None
    min_vertex_depth: int = 1
    min_edge_kmer_density: int = 2
    selected: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Colouring:
    """One registered colouring."""

    name: str
    apply: Callable[[CfaGraph, BuildContext], CfaGraph]
    available: Callable[[CfaGraph, BuildContext], bool]


_REGISTRY: dict[str, Colouring] = {}


def register(
    name: str,
    apply: Callable[[CfaGraph, BuildContext], CfaGraph],
    *,
    available: Callable[[CfaGraph, BuildContext], bool] | None = None,
) -> None:
    """Register ``name``. Registering the same name again replaces the colouring."""
    if not name or any(char.isspace() for char in name):
        raise ValueError(f"colouring name must be a non-empty token, got {name!r}")
    ready = available if available is not None else (lambda _graph, _ctx: True)
    _REGISTRY[name] = Colouring(name=name, apply=apply, available=ready)


def unregister(name: str) -> None:
    """Remove a colouring. Used by tests that install a temporary colouring."""
    _REGISTRY.pop(name, None)


def registered_names() -> tuple[str, ...]:
    """Return colouring names in registration order."""
    return tuple(_REGISTRY)


def applicable(graph: CfaGraph, ctx: BuildContext) -> tuple[Colouring, ...]:
    """Return colourings that can run, honouring an explicit selection."""
    chosen = _REGISTRY if ctx.selected is None else {
        name: _REGISTRY[name] for name in ctx.selected if name in _REGISTRY
    }
    missing = [] if ctx.selected is None else [name for name in ctx.selected if name not in _REGISTRY]
    if missing:
        from metametro.errors import ContractError

        raise ContractError([f"unknown colouring: {', '.join(missing)}"])
    return tuple(item for item in chosen.values() if item.available(graph, ctx))


def _as_built(graph: CfaGraph, _ctx: BuildContext) -> CfaGraph:
    return graph


def _has_k(graph: CfaGraph) -> bool:
    k = graph.metadata.get("k")
    return isinstance(k, int) and not isinstance(k, bool) and k > 0


def _reads_available(graph: CfaGraph, ctx: BuildContext) -> bool:
    return bool(ctx.reads) and bool(ctx.samples) and _has_k(graph)


def _read_depth(graph: CfaGraph, ctx: BuildContext) -> CfaGraph:
    assert ctx.reads is not None and ctx.samples is not None
    operation = "merge" if _already_coloured(graph) else "replace"
    return colour_by_reads(
        graph,
        ctx.reads,
        ctx.samples,
        min_vertex_depth=ctx.min_vertex_depth,
        min_edge_kmer_density=ctx.min_edge_kmer_density,
        operation=operation,
    )


def _already_coloured(graph: CfaGraph) -> bool:
    for row in list(graph.nodes) + list(graph.edges):
        if str(row.get("color_set", "")).strip():
            return True
    return False


def _install_builtins() -> None:
    if "as_built" not in _REGISTRY:
        register("as_built", _as_built)
    if "read_depth" not in _REGISTRY:
        register("read_depth", _read_depth, available=_reads_available)


_install_builtins()
