"""Optional checks that need extra software."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.optional


def test_pyg_adapter_when_installed() -> None:
    """Build a PyG Data object when torch_geometric is installed."""
    pytest.importorskip("torch_geometric")
    from metametro.converters.cgt_to_pyg import to_pyg
    from metametro.fixtures import mock_cgt

    data = to_pyg(mock_cgt())
    assert data.x.shape[0] == 4
    assert data.edge_index.shape == (2, 5)
    assert data.y.shape == (4,)
