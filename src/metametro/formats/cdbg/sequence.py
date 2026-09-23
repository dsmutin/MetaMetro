"""Split a unitig sequence into the CFA node sequences it stores."""

from __future__ import annotations

from metametro.errors import ContractError
from metametro.formats.cdbg.model import Unitig


def junction_overlaps(unitig: Unitig, k: int | None) -> list[int]:
    """Return one overlap per internal junction.

    Stored ``internal_overlaps`` are used when present. A positive de Bruijn
    ``k`` supplies ``k - 1`` when they were not stored. Any other unitig
    raises ``ContractError``.
    """
    junctions = len(unitig.members) - 1
    if junctions <= 0:
        return []
    if unitig.internal_overlaps:
        if len(unitig.internal_overlaps) != junctions:
            raise ContractError(
                [f"unitig {unitig.unitig_id} overlaps do not match its CFA members"]
            )
        return list(unitig.internal_overlaps)
    if isinstance(k, int) and not isinstance(k, bool) and k > 0:
        return [k - 1] * junctions
    raise ContractError(
        [f"cannot split unitig {unitig.unitig_id} without stored overlaps or k"]
    )


def split_unitig(sequence: str, lengths: list[int], overlaps: list[int]) -> list[str]:
    """Cut ``sequence`` into one string per CFA node length.

    ``overlaps[i]`` is the overlap between node ``i`` and node ``i + 1``.
    The pieces must consume ``sequence`` exactly.
    """
    if not lengths:
        return []
    if len(overlaps) != len(lengths) - 1:
        raise ContractError(["cannot split unitig sequence with the stored node lengths"])
    pieces = [sequence[: lengths[0]]]
    cursor = lengths[0]
    for length, overlap in zip(lengths[1:], overlaps):
        extra = length - overlap
        if overlap < 0 or extra < 0 or cursor + extra > len(sequence):
            raise ContractError(["cannot split unitig sequence with the stored node lengths"])
        if overlap == 0:
            prefix = ""
        elif len(pieces[-1]) < overlap:
            raise ContractError(["cannot split unitig sequence with the stored node lengths"])
        else:
            prefix = pieces[-1][-overlap:]
        pieces.append(prefix + sequence[cursor : cursor + extra])
        cursor += extra
    if cursor != len(sequence):
        raise ContractError(["unitig sequence length does not match mapped node lengths"])
    return pieces
