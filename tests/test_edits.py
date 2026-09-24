"""Mandatory checks for immutable CDBG edit proposals."""

from __future__ import annotations

from copy import deepcopy

import pytest

from metametro.converters.cdbg_to_cgt import cdbg_to_cgt
from metametro.edits import EditProposal, GraphEdit, apply_edit_proposal, validate_edit_proposal
from metametro.errors import ContractError
from metametro.fixtures import chain_cdbg
from metametro.formats.cdbg import annotate_cdbg
from metametro.formats.cdbg import annotate_cdbg
from metametro.identity import cgt_edge_cfa_ids, node_lineage

pytestmark = pytest.mark.mandatory


def _edit(edit_id: str, operation: str, target: str, **kwargs) -> GraphEdit:
    parameters = dict(kwargs.pop("parameters", {}))
    return GraphEdit(
        edit_id=edit_id,
        operation=operation,
        target=target,
        secondary_target=kwargs.pop("secondary_target", None),
        parameters=parameters,
        reason=kwargs.pop("reason", "test"),
        confidence=kwargs.pop("confidence", 0.5),
        model_id=kwargs.pop("model_id", "fixture"),
        model_version=kwargs.pop("model_version", "0"),
    )


def _proposal(*edits: GraphEdit, proposal_id: str = "p1") -> EditProposal:
    return EditProposal(proposal_id=proposal_id, edits=edits, source="test")


def _signature(cdbg) -> dict:
    return {
        "graph_id": cdbg.metadata.get("graph_id"),
        "provenance": deepcopy(cdbg.metadata.get("edit_provenance")),
        "unitigs": [
            (item.unitig_id, item.sequence, tuple(item.members), tuple(item.internal_edge_ids))
            for item in cdbg.unitigs
        ],
        "links": [
            (link.link_id, link.source, link.target, link.orientation, link.overlap)
            for link in cdbg.links
        ],
        "mapping": [
            (row.cfa_node_id, row.unitig_id, row.ordinal, row.length) for row in cdbg.mapping
        ],
        "annotations": [
            (layer.namespace, layer.feature, layer.target_type, tuple(map(str, layer.target_ids)), tuple(map(str, layer.values.tolist())))
            for layer in cdbg.annotations
        ],
    }


def _unitig_with(cdbg, members: set[str]):
    found = [unitig for unitig in cdbg.unitigs if set(unitig.members) == members]
    assert len(found) == 1
    return found[0]


def _provenance(cdbg) -> dict:
    return {
        "source": "test",
        "method": "annotate",
        "version": "0",
        "parameters": {},
        "parent_graph_id": cdbg.metadata["graph_id"],
        "parent_schema_version": cdbg.metadata["schema_version"],
        "parent_contract_version": cdbg.metadata["contract_version"],
    }


def test_split_at_member_boundary_keeps_cfa_ids() -> None:
    cdbg = chain_cdbg()
    parent = _unitig_with(cdbg, {"n000003", "n000004", "n000005"})
    updated = apply_edit_proposal(
        _proposal(_edit("s1", "split_unitig", parent.unitig_id, parameters={"cut_after": 0})),
        cdbg,
    )
    children = [
        unitig
        for unitig in updated.unitigs
        if unitig.unitig_id in updated.metadata["edit_provenance"]["unitig_parents"]
    ]
    assert {tuple(unitig.members) for unitig in children} == {("n000003",), ("n000004", "n000005")}
    for unitig in children:
        assert updated.metadata["edit_provenance"]["unitig_parents"][unitig.unitig_id] == [parent.unitig_id]
    assert updated.metadata["graph_id"] != cdbg.metadata["graph_id"]
    assert _signature(cdbg)["unitigs"] == _signature(chain_cdbg())["unitigs"]


def test_merge_forward_link_records_both_parents() -> None:
    cdbg = chain_cdbg()
    parent = _unitig_with(cdbg, {"n000003", "n000004", "n000005"})
    split = apply_edit_proposal(
        _proposal(_edit("s1", "split_unitig", parent.unitig_id, parameters={"cut_after": 0}), proposal_id="split"),
        cdbg,
    )
    left = _unitig_with(split, {"n000003"})
    right = _unitig_with(split, {"n000004", "n000005"})
    merged = apply_edit_proposal(
        _proposal(
            _edit("m1", "merge_unitigs", left.unitig_id, secondary_target=right.unitig_id),
            proposal_id="merge",
        ),
        split,
    )
    restored = _unitig_with(merged, {"n000003", "n000004", "n000005"})
    assert merged.metadata["edit_provenance"]["unitig_parents"][restored.unitig_id] == [
        left.unitig_id,
        right.unitig_id,
    ]
    assert restored.sequence == parent.sequence
    assert _signature(split) == _signature(split)


def test_remove_edge_and_failed_remove_leave_original() -> None:
    cdbg = chain_cdbg()
    before = _signature(cdbg)
    link_id = cdbg.links[0].link_id
    updated = apply_edit_proposal(_proposal(_edit("r1", "remove_edge", link_id)), cdbg)
    assert all(link.link_id != link_id for link in updated.links)
    assert _signature(cdbg) == before
    with pytest.raises(ContractError, match="nonexistent target"):
        apply_edit_proposal(_proposal(_edit("bad", "remove_edge", "missing")), cdbg)
    assert _signature(cdbg) == before


def test_add_edge_rejects_overlap_mismatch() -> None:
    cdbg = chain_cdbg()
    before = _signature(cdbg)
    ends = sorted(cdbg.unitigs, key=lambda unitig: unitig.unitig_id)
    with pytest.raises(ContractError, match="overlap mismatch"):
        apply_edit_proposal(
            _proposal(
                _edit(
                    "a1",
                    "add_edge",
                    ends[0].unitig_id,
                    secondary_target=ends[-1].unitig_id,
                    parameters={"orientation": "++", "overlap": 3},
                )
            ),
            cdbg,
        )
    assert _signature(cdbg) == before
    with pytest.raises(ContractError, match="invalid orientation"):
        validate_edit_proposal(
            _proposal(
                _edit(
                    "a2",
                    "add_edge",
                    ends[0].unitig_id,
                    secondary_target=ends[-1].unitig_id,
                    parameters={"orientation": "xx", "overlap": 2},
                )
            ),
            cdbg,
        )


def test_add_edge_with_matching_overlap_has_no_cfa_parent() -> None:
    cdbg = chain_cdbg()
    source = _unitig_with(cdbg, {"n000001", "n000002"})
    target = _unitig_with(cdbg, {"n000006"})
    updated = apply_edit_proposal(
        _proposal(
            _edit(
                "a1",
                "add_edge",
                source.unitig_id,
                secondary_target=target.unitig_id,
                parameters={"orientation": "++", "overlap": 2},
            )
        ),
        cdbg,
    )
    added = [link for link in updated.links if link.link_id not in {item.link_id for item in cdbg.links}]
    assert len(added) == 1
    assert updated.metadata["edit_provenance"]["link_parents"][added[0].link_id] is None


def test_annotation_label_and_suspicious_do_not_change_topology() -> None:
    cdbg = chain_cdbg()
    unitig = cdbg.unitigs[0]
    annotate_cdbg(
        cdbg,
        namespace="sample",
        feature="coverage",
        target_type="node",
        values={unitig.unitig_id: 10.0},
        dtype="float64",
        provenance=_provenance(cdbg),
    )
    annotate_cdbg(
        cdbg,
        namespace="genome",
        feature="label",
        target_type="node",
        values={item.unitig_id: "a" for item in cdbg.unitigs},
        dtype="category",
        provenance=_provenance(cdbg),
    )
    before_links = [(link.link_id, link.source, link.target) for link in cdbg.links]
    updated = apply_edit_proposal(
        _proposal(
            _edit(
                "c1",
                "change_node_annotation",
                unitig.unitig_id,
                parameters={"namespace": "sample", "feature": "coverage", "dtype": "float64", "value": 12.0},
            ),
            _edit(
                "c2",
                "reassign_label",
                unitig.unitig_id,
                parameters={"namespace": "genome", "feature": "label", "value": "b"},
            ),
            _edit("c3", "mark_suspicious", unitig.unitig_id, parameters={"target_type": "node"}),
        ),
        cdbg,
    )
    coverage = next(
        layer
        for layer in updated.annotations
        if layer.namespace == "sample" and layer.feature == "coverage"
    )
    assert dict(zip(coverage.target_ids.tolist(), coverage.values.tolist()))[unitig.unitig_id] == 12.0
    label = next(layer for layer in updated.annotations if layer.feature == "label")
    assert dict(zip(label.target_ids.tolist(), label.values.tolist()))[unitig.unitig_id] == "b"
    flag = next(layer for layer in updated.annotations if layer.feature == "suspicious")
    assert int(dict(zip(flag.target_ids.tolist(), flag.values.tolist()))[unitig.unitig_id]) == 1
    assert [(link.link_id, link.source, link.target) for link in updated.links] == before_links


def test_invalid_target_and_contradiction_do_not_change_graph() -> None:
    cdbg = chain_cdbg()
    before = _signature(cdbg)
    parent = _unitig_with(cdbg, {"n000003", "n000004", "n000005"})
    with pytest.raises(ContractError, match="nonexistent target"):
        apply_edit_proposal(_proposal(_edit("x", "split_unitig", "missing", parameters={"cut_after": 0})), cdbg)
    with pytest.raises(ContractError, match="contradictory"):
        apply_edit_proposal(
            _proposal(
                _edit("s", "split_unitig", parent.unitig_id, parameters={"cut_after": 0}),
                _edit("d", "remove_node", parent.unitig_id),
            ),
            cdbg,
        )
    assert _signature(cdbg) == before


def test_two_applies_keep_cfa_ids_and_parent_chain() -> None:
    cdbg = chain_cdbg()
    parent = _unitig_with(cdbg, {"n000003", "n000004", "n000005"})
    first = apply_edit_proposal(
        _proposal(_edit("s1", "split_unitig", parent.unitig_id, parameters={"cut_after": 0}), proposal_id="first"),
        cdbg,
    )
    right = _unitig_with(first, {"n000004", "n000005"})
    second = apply_edit_proposal(
        _proposal(_edit("s2", "split_unitig", right.unitig_id, parameters={"cut_after": 0}), proposal_id="second"),
        first,
    )
    assert second.metadata["edit_provenance"]["parent_graph_id"] == first.metadata["graph_id"]
    assert second.metadata["edit_provenance"]["history"]["parent_graph_id"] == cdbg.metadata["graph_id"]
    members = {node_id for unitig in second.unitigs for node_id in unitig.members}
    assert members == {node_id for unitig in cdbg.unitigs for node_id in unitig.members}


def test_cfa_cdbg_cgt_proposal_round_trip_keeps_edge_alignment() -> None:
    cdbg = chain_cdbg()
    parent = _unitig_with(cdbg, {"n000003", "n000004", "n000005"})
    features = {link.link_id: [float(index)] for index, link in enumerate(cdbg.links)}
    before = cdbg_to_cgt(cdbg, edge_features=features)
    updated = apply_edit_proposal(
        _proposal(_edit("s1", "split_unitig", parent.unitig_id, parameters={"cut_after": 1})),
        cdbg,
    )
    after = cdbg_to_cgt(
        updated,
        edge_features={link.link_id: [1.0] for link in updated.links},
    )
    assert cgt_edge_cfa_ids(updated) == tuple(link.link_id for _index, link in _csr_links(updated))
    assert len(cgt_edge_cfa_ids(updated)) == after.num_edges
    lineage = node_lineage(after, 0)
    assert lineage.cfa_node_ids
    assert before.num_nodes == 3
    assert _signature(cdbg)["unitigs"] == _signature(chain_cdbg())["unitigs"]


def test_annotated_remove_is_rejected_before_apply() -> None:
    cdbg = chain_cdbg()
    unitig = cdbg.unitigs[0]
    link_id = next(
        link.link_id
        for link in cdbg.links
        if link.source == unitig.unitig_id or link.target == unitig.unitig_id
    )
    annotate_cdbg(
        cdbg,
        namespace="sample",
        feature="coverage",
        target_type="node",
        values={unitig.unitig_id: 1.0},
        dtype="float64",
        provenance=_provenance(cdbg),
    )
    with pytest.raises(ContractError, match="node annotation"):
        validate_edit_proposal(_proposal(_edit("d", "remove_node", unitig.unitig_id)), cdbg)
    annotate_cdbg(
        cdbg,
        namespace="sample",
        feature="support",
        target_type="edge",
        values={link_id: 1.0},
        dtype="float64",
        provenance=_provenance(cdbg),
    )
    before = _signature(cdbg)
    with pytest.raises(ContractError, match="edge annotation"):
        validate_edit_proposal(_proposal(_edit("e", "remove_edge", link_id)), cdbg)
    assert _signature(cdbg) == before


def _csr_links(cdbg):
    from metametro.converters.cdbg_to_cgt import csr_link_order

    return [(index, cdbg.links[index]) for index in csr_link_order(cdbg)]
