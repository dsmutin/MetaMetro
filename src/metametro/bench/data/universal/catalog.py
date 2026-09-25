"""Pinned community and external benchmarks.

Accession tables are the generation contract. Strong100 is not in this list.
A contract build writes the pin and the lognormal abundance table. It does not
invent an assembly graph. ``execute`` runs the NCBI, Samovar, and MEGAHIT
tools when they are on ``PATH``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from metametro.bench.data.universal.community import iss_key, load_pairs, lognormal_read_counts
from metametro.bench.paths import repo_root
from metametro.bench.spec import BenchSpec
from metametro.errors import ContractError
from metametro.tables import write_yaml

PINS = Path(__file__).resolve().parents[1] / "pins"

_COMMUNITY_ROWS = (
    ("4domain_family_100", "high100", 100, 4, "family", 100_000, ""),
    ("4domain_family_100_half", "half100half", 50, 4, "family", 100_000, "4domain_family_100"),
    ("4domain_family_100_x10", "high100_enriched", 100, 4, "family", 1_000_000, "4domain_family_100"),
    ("4domain_family_100_half_x10", "half100half_enriched", 50, 4, "family", 1_000_000, "4domain_family_100_half"),
    ("3domain_genus_75", "low75", 75, 3, "genus", 100_000, ""),
    ("3domain_genus_75_half", "low75half", 38, 3, "genus", 100_000, "3domain_genus_75"),
    ("3domain_genus_75_x10", "low75_enriched", 75, 3, "genus", 1_000_000, "3domain_genus_75"),
    ("3domain_genus_75_half_x10", "low75half_enriched", 38, 3, "genus", 1_000_000, "3domain_genus_75_half"),
    ("bacteria_species_20_heldout", "heldout_genera", 20, 1, "species", 100_000, ""),
    ("bacteria_strain_10", "half_strains", 10, 1, "species", 100_000, "bacteria_species_20_heldout"),
)


def community_specs() -> dict[str, BenchSpec]:
    """Return the renamed NCBI community benchmarks."""
    specs = {}
    for name, legacy, pairs, domains, rank, reads, parent in _COMMUNITY_ROWS:
        specs[name] = BenchSpec(
            name=name,
            assembler="megahit",
            properties=f"reads{reads}",
            summary=(
                f"{pairs} pairs, {domains} domain group(s), scored at {rank}, "
                f"{reads} paired fragments. Was {legacy}."
            ),
            scoring=("profiling", "assembly"),
            kind="community",
            legacy_names=(legacy,),
            score_rank=rank,
            total_reads=reads,
            pair_count=pairs,
            domain_count=domains,
            k=0,
            parent=parent,
        )
    return specs


def external_specs() -> dict[str, BenchSpec]:
    """Return benchmarks whose graphs need an external tool, plus their contracts."""
    phage = (
        "Five RefSeq enterobacteria phages (T1, T3, T4, T5, T7). "
        "The old name phage_x10 is a 10× read budget, not ten species."
    )
    return {
        "phage_species_5": BenchSpec(
            name="phage_species_5",
            assembler="megahit",
            properties="reads400_k21",
            summary=phage + " 400 paired fragments.",
            scoring=("assembly", "assembly_binning"),
            kind="external",
            legacy_names=("phage_baseline",),
            total_reads=400,
            pair_count=5,
            k=21,
        ),
        "phage_species_5_x10": BenchSpec(
            name="phage_species_5_x10",
            assembler="megahit",
            properties="reads4000_k21",
            summary=phage + " 4000 paired fragments (the former phage_x10 run).",
            scoring=("assembly", "assembly_binning"),
            kind="external",
            legacy_names=("phage_x10", "phage_10"),
            total_reads=4000,
            pair_count=5,
            k=21,
        ),
        "roxel": BenchSpec(
            name="roxel",
            assembler="sfnetworks",
            properties="k21",
            summary="sfnetworks roxel street graph. Node sequences are a repeat tape, not DNA.",
            scoring=(),
            kind="external",
            k=21,
        ),
        "spb_ground_transit": BenchSpec(
            name="spb_ground_transit",
            assembler="gtfs",
            properties="merge60m",
            summary="Saint Petersburg ground transit from the public ORGP GTFS feed. Stops within 60 m that share a name are merged.",
            scoring=(),
            kind="external",
        ),
    }


def pin_dir(name: str) -> Path:
    """Return the directory that holds ``accessions.tsv`` for a community."""
    path = PINS / name
    if not (path / "accessions.tsv").is_file():
        raise ContractError([f"missing accession pin for {name}"])
    return path


def write_community_contract(spec: BenchSpec, outdir: Path) -> None:
    """Copy the accession pin and write the lognormal abundance table."""
    rows = load_pairs(pin_dir(spec.name) / "accessions.tsv")
    pair_ids = []
    seen: set[str] = set()
    for row in rows:
        if row["pair_id"] not in seen:
            seen.add(row["pair_id"])
            pair_ids.append(row["pair_id"])
    if len(pair_ids) != spec.pair_count:
        raise ContractError(
            [f"{spec.name} pin has {len(pair_ids)} pairs, expected {spec.pair_count}"]
        )
    domains = sorted({row["domain"] for row in rows if row.get("domain")})
    if spec.domain_count > 1 and len(domains) != spec.domain_count:
        raise ContractError(
            [f"{spec.name} pin has domains {domains}, expected {spec.domain_count}"]
        )
    outdir.mkdir(parents=True, exist_ok=True)
    contract = outdir / "contract"
    contract.mkdir(parents=True, exist_ok=True)
    (contract / "accessions.tsv").write_text(
        (pin_dir(spec.name) / "accessions.tsv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    sim = [row for row in rows if row["role"] == "sim"]
    counts = lognormal_read_counts(len(sim), spec.total_reads, 42, mu=0.0, sigma=1.5)
    lines = ["taxid,N_sample"]
    for row, count in zip(sim, counts):
        lines.append(f"{row['accession']},{count}")
    truth = outdir / "ground_truth"
    truth.mkdir(parents=True, exist_ok=True)
    (truth / "abundance.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (truth / "abundance_parameters.txt").write_text(
        f"total_reads={spec.total_reads}\nseed=42\nmu=0\nsigma=1.5\n"
        "distribution=random.Random.lognormvariate\n"
        "unit=paired fragments passed to samovar --total_reads\n"
        "header_note=column taxid stores the full assembly accession; Samovar splits on the first dot\n",
        encoding="utf-8",
    )
    write_yaml(
        contract / "parameters.yaml",
        {
            "bench": spec.name,
            "legacy_name": spec.legacy_names[0] if spec.legacy_names else "",
            "assembler": spec.assembler,
            "score_rank": spec.score_rank,
            "total_reads": spec.total_reads,
            "pair_count": spec.pair_count,
            "seed": 42,
            "sigma": 1.5,
            "parent": spec.parent,
            "iss_keys": [iss_key(row["accession"]) for row in sim],
        },
    )


def write_external_contract(spec: BenchSpec, outdir: Path) -> None:
    """Write the tool contract. This does not download and does not invent a graph."""
    outdir.mkdir(parents=True, exist_ok=True)
    contract = outdir / "contract"
    contract.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "bench": spec.name,
        "assembler": spec.assembler,
        "properties": spec.properties,
        "total_reads": spec.total_reads,
        "k": spec.k,
    }
    if spec.name.startswith("phage_species_5"):
        payload["genomes"] = ["T1", "T3", "T4", "T5", "T7"]
        payload["accessions"] = [
            "NC_005833.1",
            "NC_003298.1",
            "NC_000866.4",
            "NC_005859.1",
            "NC_001604.1",
        ]
        payload["genome_dir"] = "data/raw/genomes"
    if spec.name == "roxel":
        payload["source"] = "sfnetworks::as_sfnetwork(sfnetworks::roxel)"
        payload["program"] = "Rscript"
    if spec.name == "spb_ground_transit":
        payload["feed_url"] = "https://transport.orgp.spb.ru/Portal/transport/internalapi/gtfs/feed.zip"
        payload["merge_metres"] = 60
    write_yaml(contract / "parameters.yaml", payload)


def missing_programs(names: tuple[str, ...]) -> list[str]:
    """Return required executables that are not on ``PATH``."""
    return [name for name in names if shutil.which(name) is None]


def require_programs(spec: BenchSpec, names: tuple[str, ...]) -> None:
    """Stop before a download when a required program is absent."""
    missing = missing_programs(names)
    if missing:
        raise ContractError(
            [f"{spec.name} needs these programs on PATH before a build can start: {', '.join(missing)}"]
        )


def repo_data_root() -> Path:
    """Return the repository root used for raw downloads."""
    return repo_root()
