"""Check that a benchbuild directory can be loaded as a model input.

Contract builds have no graph. Built graphs must load as CFA, CDBG, and CGT,
keep topology when a colour namespace is selected, and keep evaluation targets
out of those graphs.
"""

from __future__ import annotations

from pathlib import Path

from metametro.bench.leakage import assert_no_target_leak
from metametro.bench.paths import default_outdir
from metametro.bench.registry import list_specs
from metametro.bench.select import load_bench_cdbg, load_bench_cgt, namespaces_for
from metametro.errors import ContractError
from metametro.formats.cdbg.io import load_cdbg
from metametro.formats.cfa.io import load_cfa
from metametro.formats.cgt.io import load_cgt
from metametro.identity import assert_cgt_matches_cdbg
from metametro.tables import read_yaml


def check_bench_directory(outdir: Path) -> str:
    """Validate one ``benchbuild`` directory. Return its manifest status.

    ``contract`` means the pin was written and no assembly graph was invented.
    ``built`` means the three graphs load and colour selection keeps topology.
    """
    path = Path(outdir)
    manifest_path = path / "manifest.yaml"
    if not manifest_path.is_file():
        raise ContractError([f"missing manifest.yaml in {path}"])
    manifest = read_yaml(manifest_path)
    status = str(manifest.get("status", ""))
    if status == "contract":
        if not (path / "contract").is_dir():
            raise ContractError([f"{path} is a contract build without contract/"])
        if (path / "cfa").is_dir() or (path / "cdbg").is_dir() or (path / "cgt").is_dir():
            raise ContractError([f"{path} is a contract build but contains a graph"])
        return status
    if status != "built":
        raise ContractError([f"{path} has status {status!r}; expected built or contract"])
    if not (path / "identity.sha256").is_file():
        raise ContractError([f"{path} is built but has no identity.sha256"])
    cfa = load_cfa(path / "cfa")
    cdbg = load_cdbg(path / "cdbg")
    cgt = load_cgt(path / "cgt")
    assert_no_target_leak(cfa)
    if cgt.num_nodes != len(cdbg.unitigs):
        raise ContractError(
            [f"{path} CGT has {cgt.num_nodes} nodes and CDBG has {len(cdbg.unitigs)} unitigs"]
        )
    assert_cgt_matches_cdbg(cgt, cdbg)
    names = [str(item) for item in manifest.get("colourings") or [] if item != "as_built"]
    namespaces = namespaces_for(names)
    if namespaces:
        selected = load_bench_cdbg(path, namespaces=namespaces)
        if len(selected.unitigs) != len(cdbg.unitigs) or len(selected.links) != len(cdbg.links):
            raise ContractError([f"colour selection changed topology in {path}"])
        retensorised = load_bench_cgt(path, namespaces=namespaces)
        if retensorised.num_nodes != cgt.num_nodes:
            raise ContractError([f"colour selection changed the CGT node count in {path}"])
    return status


def check_registry(root: Path | None = None) -> list[tuple[str, str]]:
    """Check every registered benchmark under ``data/bench`` of ``root``.

    ``root`` defaults to the repository. Each directory must already exist.
    This does not build or download.
    """
    results: list[tuple[str, str]] = []
    missing: list[str] = []
    for spec in list_specs():
        path = default_outdir(spec, root=root)
        if not (path / "manifest.yaml").is_file():
            missing.append(spec.name)
            continue
        results.append((spec.name, check_bench_directory(path)))
    if missing:
        raise ContractError(
            ["benchbuild output is missing for: " + ", ".join(missing)]
        )
    return results
