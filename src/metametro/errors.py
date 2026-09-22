"""Errors raised when a graph contract is violated."""

from __future__ import annotations


class ContractError(ValueError):
    """One or more contract checks failed.

    ``errors`` lists every problem found in that check. The message joins them
    so a caller can match a single failure mode in tests.
    """

    def __init__(self, errors: list[str]) -> None:
        if not errors:
            raise ValueError("ContractError requires at least one error")
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))
