"""Held-out community design: accessions, lognormal reads, FASTG edges, scores.

Samovar looks up an abundance-table taxid after ``split('.')``, so an assembly
accession ``GCF_000005845.2`` is stored for InSilicoSeq as ``GCF_000005845``.
Database FASTA files keep the full accession so Samovar can resolve the NCBI
taxid. The two names are the same sequence, not a second genome.
"""

from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path


def iss_key(accession: str) -> str:
    """Return the Samovar abundance key for an assembly accession."""
    text = accession.strip()
    if not text or "." not in text:
        raise ValueError(f"expected an accession with a version, got {accession!r}")
    return text.split(".", 1)[0]


def load_pairs(path: Path) -> list[dict[str, str]]:
    """Load the pinned accession table. Every pair needs one sim row and one db row."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"empty accession table: {path}")
    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        values = line.split("\t")
        if len(values) != len(header):
            raise ValueError(f"accession row has {len(values)} columns, expected {len(header)}")
        rows.append(dict(zip(header, values)))
    by_pair: dict[str, list[str]] = {}
    keys = []
    for row in rows:
        by_pair.setdefault(row["pair_id"], []).append(row["role"])
        if row["role"] == "sim":
            keys.append(iss_key(row["accession"]))
    problems = []
    for pair_id, roles in sorted(by_pair.items()):
        if sorted(roles) != ["db", "sim"]:
            problems.append(f"{pair_id} roles {roles}")
    if len(keys) != len(set(keys)):
        problems.append("sim accessions collide after dropping the version")
    if problems:
        raise ValueError("accession table: " + "; ".join(problems))
    return rows


def every_other_strain(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep pairs at even positions in first-seen order, including both roles.

    An odd strain count keeps one more pair than it drops. ``half_strains``
    rejects that case so a named half stays exact.
    """
    order: list[str] = []
    seen: set[str] = set()
    for row in rows:
        pair_id = row["pair_id"]
        if pair_id not in seen:
            seen.add(pair_id)
            order.append(pair_id)
    if not order:
        raise ValueError("every_other_strain needs at least one strain")
    keep = set(order[::2])
    return [row for row in rows if row["pair_id"] in keep]


def half_strains(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep every other strain in the order pairs first appear.

    Twenty strains become ten. Both the sim row and the db row of a kept pair
    are returned. An odd strain count is rejected so the half is exact.
    """
    order = []
    seen: set[str] = set()
    for row in rows:
        if row["pair_id"] not in seen:
            seen.add(row["pair_id"])
            order.append(row["pair_id"])
    if len(order) % 2:
        raise ValueError(f"half_strains needs an even strain count, got {len(order)}")
    return every_other_strain(rows)


def lognormal_read_counts(
    n_genomes: int,
    total_reads: int,
    seed: int,
    *,
    mu: float = 0.0,
    sigma: float = 1.5,
) -> list[int]:
    """Split ``total_reads`` across genomes with a lognormal draw.

    ``mu`` and ``sigma`` are the parameters of ``random.Random.lognormvariate``.
    Every genome receives at least one read. The integers sum to ``total_reads``.
    """
    if n_genomes < 1:
        raise ValueError("n_genomes must be positive")
    if total_reads < n_genomes:
        raise ValueError("total_reads must be at least one read per genome")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    rng = random.Random(seed)
    weights = [rng.lognormvariate(mu, sigma) for _ in range(n_genomes)]
    counts = _largest_remainder(weights, total_reads)
    guard = 0
    while min(counts) < 1:
        donor = max(range(n_genomes), key=lambda index: counts[index])
        receiver = counts.index(0)
        if counts[donor] <= 1:
            raise ValueError("cannot give every genome a read")
        counts[donor] -= 1
        counts[receiver] += 1
        guard += 1
        if guard > n_genomes:
            raise ValueError("lognormal allocation did not cover every genome")
    if sum(counts) != total_reads:
        raise ValueError("lognormal allocation does not sum to total_reads")
    return counts


def _largest_remainder(weights: list[float], total: int) -> list[int]:
    scale = sum(weights)
    if scale <= 0:
        raise ValueError("weights must be positive")
    raw = [weight / scale * total for weight in weights]
    counts = [int(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(raw)), key=lambda index: (raw[index] - counts[index], -index), reverse=True)
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def sequence_digest(path: Path) -> str:
    """SHA-256 of uppercase sequence characters, ignoring headers and whitespace."""
    digest = hashlib.sha256()
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(">"):
                continue
            digest.update(re.sub(r"[^A-Za-z]", "", line).upper().encode("ascii"))
    return digest.hexdigest()


def parse_fastg(text: str) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Parse FASTG nodes and the neighbour links written in each header.

    A header ``>NAME:NEIGH,NEIGH';`` lists overlaps. A trailing quote is a
    reverse-complement link. The sequence is the following lines up to the
    next header. Node ids are the header token before the first colon.
    """
    edges: list[dict[str, str]] = []
    sequences: dict[str, str] = {}
    header = ""
    chunks: list[str] = []

    def flush() -> None:
        if not header:
            return
        body = header[1:] if header.startswith(">") else header
        body = body.strip().rstrip(";")
        name, _, neighbours = body.partition(":")
        node_id = name.strip().split()[0]
        if not node_id:
            raise ValueError(f"FASTG header has no node id: {header}")
        sequences[node_id] = "".join(chunks).replace("\n", "").strip()
        for piece in neighbours.split(","):
            token = piece.strip()
            if not token:
                continue
            orientation = "+-" if token.endswith("'") else "++"
            target = token[:-1] if token.endswith("'") else token
            target = target.strip()
            if not target:
                continue
            edges.append(
                {
                    "edge_id": f"{node_id}->{target}:{orientation}",
                    "source": node_id,
                    "target": target,
                    "orientation": orientation,
                    "weight": "1",
                }
            )

    for line in text.splitlines():
        if line.startswith(">"):
            flush()
            header = line.strip()
            chunks = []
        else:
            chunks.append(line.strip())
    flush()
    return edges, sequences


def highest_megahit_contigs(names: list[str]) -> str | None:
    """Pick ``k<int>.contigs.fa``, not ``k<int>.final.contigs.fa``.

    ``contig2fastg`` reads the plain ``k*.contigs.fa`` file. A lexical sort of
    every ``k*.contigs.fa`` name selects ``k99.final.contigs.fa`` ahead of
    ``k141.contigs.fa`` and the toolkit then writes an empty FASTG.
    """
    pattern = re.compile(r"k(\d+)\.contigs\.fa$")
    hits = []
    for name in names:
        match = pattern.search(Path(name).name)
        if match:
            hits.append((int(match.group(1)), name))
    if not hits:
        return None
    return max(hits)[1]


def megahit_coverage(header: str) -> float | None:
    """Return MEGAHIT coverage from ``multi=`` or from a FASTG ``_cov_`` field."""
    match = re.search(r"(?:\bmulti=|_cov_)([0-9]+(?:\.[0-9]+)?)", header)
    if not match:
        return None
    return float(match.group(1))


def assembly_graph_from_fastg(text: str) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Keep forward FASTG records and the overlap links among them.

    MEGAHIT writes the reverse record as the same name with a trailing quote.
    Those sequences repeat the forward contig, so they are not extra nodes.
    """
    edges, sequences = parse_fastg(text)

    def forward_id(node_id: str) -> str:
        return node_id[:-1] if node_id.endswith("'") else node_id

    forward = {node_id: sequence for node_id, sequence in sequences.items() if not node_id.endswith("'")}
    kept = []
    seen = set()
    for edge in edges:
        source = forward_id(edge["source"])
        target = forward_id(edge["target"])
        if source not in forward or target not in forward or source == target:
            continue
        key = (source, target)
        if key in seen:
            continue
        seen.add(key)
        kept.append(
            {
                "edge_id": f"{source}->{target}",
                "source": source,
                "target": target,
                "orientation": edge["orientation"],
                "weight": "1",
            }
        )
    return kept, forward


COARSE_RANKS = frozenset(
    {"genus", "family", "order", "class", "phylum", "kingdom", "superkingdom", "domain"}
)


def rank_taxon(taxon_id: int, parents: dict[int, int], ranks: dict[int, str], target: str) -> int | None:
    """Walk NCBI parents to ``target`` (``species`` or ``family``).

    A start taxon already coarser than ``target`` returns None. A cycle returns
    None rather than looping.
    """
    if target not in {"species", "family"}:
        raise ValueError(f"unsupported rank {target}")
    # Family scoring climbs through genus. Species scoring stops at genus.
    if target == "species":
        above = COARSE_RANKS
    else:
        above = frozenset(
            {"order", "class", "phylum", "kingdom", "superkingdom", "domain"}
        )
    current = taxon_id
    seen: set[int] = set()
    while current and current not in seen:
        seen.add(current)
        rank = ranks.get(current, "")
        if rank == target:
            return current
        if rank in above:
            return None
        parent = parents.get(current)
        if parent is None or parent == current:
            return None
        current = parent
    return None


def species_taxon(taxon_id: int, parents: dict[int, int], ranks: dict[int, str]) -> int | None:
    """Walk NCBI parents to a species, skipping strain and no-rank nodes.

    A start taxon that is already genus or coarser returns None. A cycle returns
    None rather than looping.
    """
    current = taxon_id
    seen: set[int] = set()
    while current and current not in seen:
        seen.add(current)
        rank = ranks.get(current, "")
        if rank == "species":
            return current
        if rank in COARSE_RANKS:
            return None
        parent = parents.get(current)
        if parent is None or parent == current:
            return None
        current = parent
    return None


def r_squared(pearson: float | None) -> float | None:
    """Square of the Pearson correlation. The sign of the correlation is discarded."""
    if pearson is None:
        return None
    return pearson * pearson


def presence_f1(predicted: dict[int, float], truth: dict[int, float], *, minimum: float = 0.0) -> dict[str, float]:
    """Set F1 for taxa whose abundance is above ``minimum``. Taxon 0 is ignored."""
    truth_ids = {taxon_id for taxon_id, value in truth.items() if taxon_id != 0 and value > minimum}
    pred_ids = {taxon_id for taxon_id, value in predicted.items() if taxon_id != 0 and value > minimum}
    tp = len(truth_ids & pred_ids)
    fp = len(pred_ids - truth_ids)
    fn = len(truth_ids - pred_ids)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": float(tp), "fp": float(fp), "fn": float(fn)}
