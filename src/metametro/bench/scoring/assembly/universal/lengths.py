"""Assembly length summaries. These do not invent a genome fraction."""

from __future__ import annotations


def n50(lengths: list[int]) -> int:
    """Length of the contig at which cumulative length reaches half the total.

    An empty list is an error. Lengths must be positive.
    """
    if not lengths:
        raise ValueError("n50 requires at least one length")
    if any(length <= 0 for length in lengths):
        raise ValueError("contig lengths must be positive")
    ordered = sorted(lengths, reverse=True)
    total = sum(ordered)
    half = total / 2
    covered = 0
    for length in ordered:
        covered += length
        if covered >= half:
            return length
    return ordered[-1]
