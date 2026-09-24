"""Immutable graph edit proposals.

A model that reads a CGT does not write the CDBG or the CFA. It builds an
``EditProposal``. ``apply_edit_proposal`` validates that proposal and returns
a new CDBG. The input graph is not modified.

New unitig ids name compacted pieces. They are not biological ids. Each new
unitig records the unitig id it was split or merged from. CFA node ids on
those pieces stay the original CFA ids.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from metametro.errors import ContractError
from metametro.formats.cdbg.annotations import annotate_cdbg
from metametro.formats.cdbg.model import Cdbg, Link, Unitig
from metametro.formats.cdbg.sequence import junction_overlaps, split_unitig
from metametro.formats.cdbg.validator import validate_cdbg
from metametro.formats.cfa.model import ORIENTATIONS


OPERATIONS = frozenset(
    {
        "split_unitig",
        "merge_unitigs",
        "remove_node",
        "remove_edge",
        "add_edge",
        "change_node_annotation",
        "change_edge_annotation",
        "reassign_label",
        "mark_suspicious",
    }
)
@dataclass(frozen=True)
class GraphEdit:
    """One proposed change. ``target`` is a CDBG unitig id or link id."""

    edit_id: str
    operation: str
    target: str
    secondary_target: str | None
    parameters: dict[str, Any]
    reason: str
    confidence: float
    model_id: str
    model_version: str


@dataclass(frozen=True)
class EditProposal:
    """An immutable list of edits. This is not a graph format."""

    proposal_id: str
    edits: tuple[GraphEdit, ...]
    source: str


def validate_edit_proposal(proposal: EditProposal, cdbg: Cdbg) -> None:
    """Raise ``ContractError`` when ``proposal`` cannot be applied to ``cdbg``.

    The graph is not modified. Checks cover missing targets, duplicate ids,
    contradictory operations, overlap and orientation, cuts that are not CFA
    member boundaries, and edits that would drop provenance.
    """
    validate_cdbg(cdbg)
    errors = _proposal_errors(proposal, cdbg)
    if errors:
        raise ContractError(errors)


def validate_edit_application(proposal: EditProposal, before: Cdbg, after: Cdbg) -> None:
    """Raise ``ContractError`` when ``after`` does not record ``proposal``.

    ``before`` is the graph that was proposed against. ``after`` must name a
    different ``graph_id``, point at that parent, and list a non-empty source
    unitig for every unitig the proposal created.
    """
    validate_cdbg(before)
    validate_cdbg(after)
    errors: list[str] = []
    parent_id = str(before.metadata.get("graph_id", ""))
    child_id = str(after.metadata.get("graph_id", ""))
    if child_id == "" or child_id == parent_id:
        errors.append("edit application did not create a new graph_id")
    provenance = after.metadata.get("edit_provenance")
    if not isinstance(provenance, dict):
        errors.append("edit application is missing edit_provenance")
        raise ContractError(errors)
    if provenance.get("parent_graph_id") != parent_id:
        errors.append("edit_provenance parent_graph_id does not match the input graph")
    if provenance.get("proposal_id") != proposal.proposal_id:
        errors.append("edit_provenance proposal_id does not match the proposal")
    if list(provenance.get("edit_ids", [])) != [edit.edit_id for edit in proposal.edits]:
        errors.append("edit_provenance edit_ids do not match the proposal")
    parents = provenance.get("unitig_parents")
    if not isinstance(parents, dict):
        errors.append("edit_provenance unitig_parents is missing")
        parents = {}
    before_ids = {unitig.unitig_id for unitig in before.unitigs}
    after_ids = {unitig.unitig_id for unitig in after.unitigs}
    for unitig_id in sorted(after_ids - before_ids):
        if unitig_id not in parents:
            errors.append(f"new unitig {unitig_id} has no source unitig")
    for unitig_id, sources in parents.items():
        if not isinstance(sources, list) or not sources:
            errors.append(f"unitig {unitig_id} has no source unitig")
            continue
        if unitig_id not in after_ids:
            errors.append(f"unitig {unitig_id} is not on the edited graph")
        missing = [source for source in sources if source not in before_ids]
        if missing:
            errors.append(f"unitig {unitig_id} names a source that is not on the parent graph")
    link_parents = provenance.get("link_parents")
    if not isinstance(link_parents, dict):
        errors.append("edit_provenance link_parents is missing")
    else:
        for edit in proposal.edits:
            if edit.operation != "add_edge":
                continue
            link_id = _added_link_id(proposal, edit)
            if link_id not in link_parents or link_parents[link_id] is not None:
                errors.append(f"added link {link_id} claims a CFA edge id")
    if errors:
        raise ContractError(errors)


def apply_edit_proposal(proposal: EditProposal, cdbg: Cdbg) -> Cdbg:
    """Return a new CDBG with ``proposal`` applied.

    ``cdbg`` is not modified. An invalid proposal raises ``ContractError``
    before any copy is edited. A failure while building the result also
    leaves ``cdbg`` unchanged and does not return a partial graph.
    """
    validate_edit_proposal(proposal, cdbg)
    edited = deepcopy(cdbg)
    parent_id = str(edited.metadata.get("graph_id", ""))
    previous = edited.metadata.get("edit_provenance")
    new_id = f"{parent_id}__{proposal.proposal_id}"
    edited.metadata["graph_id"] = new_id
    edited.metadata["edit_provenance"] = {
        "parent_graph_id": parent_id,
        "proposal_id": proposal.proposal_id,
        "edit_ids": [edit.edit_id for edit in proposal.edits],
        "unitig_parents": {},
        "link_parents": {},
        "history": previous,
    }
    _retarget_annotation_graph_id(edited, new_id)
    for edit in proposal.edits:
        if edit.operation == "remove_edge":
            _remove_edge(edited, edit)
        elif edit.operation == "remove_node":
            _remove_node(edited, edit)
    for edit in proposal.edits:
        if edit.operation == "split_unitig":
            _split_unitig(edited, proposal, edit)
        elif edit.operation == "merge_unitigs":
            _merge_unitigs(edited, proposal, edit)
        elif edit.operation == "add_edge":
            _add_edge(edited, proposal, edit)
    for edit in proposal.edits:
        if edit.operation in {"change_node_annotation", "change_edge_annotation", "reassign_label", "mark_suspicious"}:
            _apply_annotation_edit(edited, proposal, edit)
    validate_cdbg(edited)
    validate_edit_application(proposal, cdbg, edited)
    return edited


def _proposal_errors(proposal: EditProposal, cdbg: Cdbg) -> list[str]:
    errors: list[str] = []
    if not isinstance(proposal, EditProposal):
        return ["edit proposal must be an EditProposal"]
    if not isinstance(proposal.proposal_id, str) or proposal.proposal_id == "":
        errors.append("proposal_id is required")
    if not isinstance(proposal.source, str) or proposal.source == "":
        errors.append("proposal source is required")
    if not isinstance(proposal.edits, tuple) or not proposal.edits:
        errors.append("edit proposal has no edits")
        return errors
    seen: set[str] = set()
    for edit in proposal.edits:
        errors.extend(_edit_shape_errors(edit))
        if isinstance(edit, GraphEdit) and edit.edit_id in seen:
            errors.append(f"duplicate edit_id: {edit.edit_id}")
        if isinstance(edit, GraphEdit):
            seen.add(edit.edit_id)
    if errors:
        return errors
    unitigs = {unitig.unitig_id: unitig for unitig in cdbg.unitigs}
    links = {link.link_id: link for link in cdbg.links}
    consumed_unitigs: dict[str, str] = {}
    consumed_links: dict[str, str] = {}
    for edit in proposal.edits:
        if edit.operation == "split_unitig":
            _claim(consumed_unitigs, edit.target, edit.edit_id, errors)
        elif edit.operation == "merge_unitigs":
            _claim(consumed_unitigs, edit.target, edit.edit_id, errors)
            if edit.secondary_target:
                _claim(consumed_unitigs, edit.secondary_target, edit.edit_id, errors)
        elif edit.operation == "remove_node":
            _claim(consumed_unitigs, edit.target, edit.edit_id, errors)
    for edit in proposal.edits:
        if edit.operation == "remove_edge":
            _claim(consumed_links, edit.target, edit.edit_id, errors)
        elif edit.operation == "merge_unitigs":
            link = _merge_link(cdbg, edit, unitigs, links, errors)
            if link is not None:
                _claim(consumed_links, link.link_id, edit.edit_id, errors)
    for edit in proposal.edits:
        errors.extend(_target_errors(edit, cdbg, unitigs, links, consumed_unitigs, consumed_links))
    errors.extend(_apply_time_errors(proposal, cdbg, unitigs))
    return errors


def _edit_shape_errors(edit: GraphEdit) -> list[str]:
    if not isinstance(edit, GraphEdit):
        return ["edit must be a GraphEdit"]
    errors: list[str] = []
    if not isinstance(edit.edit_id, str) or edit.edit_id == "":
        errors.append("edit_id is required")
    if edit.operation not in OPERATIONS:
        errors.append(f"unknown edit operation: {edit.operation}")
    if not isinstance(edit.target, str) or edit.target == "":
        errors.append("edit target is required")
    if edit.secondary_target is not None and not isinstance(edit.secondary_target, str):
        errors.append(f"edit {edit.edit_id} secondary_target must be a string")
    if not isinstance(edit.parameters, dict):
        errors.append(f"edit {edit.edit_id} parameters must be a mapping")
    if not isinstance(edit.reason, str) or edit.reason == "":
        errors.append(f"edit {edit.edit_id} reason is required")
    if isinstance(edit.confidence, bool) or not isinstance(edit.confidence, (int, float)):
        errors.append(f"edit {edit.edit_id} confidence must be a float in [0, 1]")
    elif not 0.0 <= float(edit.confidence) <= 1.0:
        errors.append(f"edit {edit.edit_id} confidence must be a float in [0, 1]")
    if not isinstance(edit.model_id, str) or edit.model_id == "":
        errors.append(f"edit {edit.edit_id} model_id is required")
    if not isinstance(edit.model_version, str) or edit.model_version == "":
        errors.append(f"edit {edit.edit_id} model_version is required")
    if edit.operation == "add_edge" and edit.parameters.get("cfa_edge_id") is not None:
        errors.append(f"edit {edit.edit_id} claims a CFA edge id for a new link")
    return errors


def _claim(owner: dict[str, str], key: str, edit_id: str, errors: list[str]) -> None:
    previous = owner.get(key)
    if previous is not None and previous != edit_id:
        errors.append(f"contradictory edits {previous} and {edit_id} both use {key}")
    owner[key] = edit_id


def _apply_time_errors(proposal: EditProposal, cdbg: Cdbg, unitigs: dict[str, Unitig]) -> list[str]:
    """Checks that used to run only inside apply, so validate and apply agree."""
    errors: list[str] = []
    for edit in proposal.edits:
        if edit.operation == "remove_edge" and _edge_annotation_targets(cdbg, edit.target):
            errors.append(f"edit {edit.edit_id} cannot remove a link that has an edge annotation")
        elif edit.operation == "remove_node" and _node_annotation_targets(cdbg, edit.target):
            errors.append(f"edit {edit.edit_id} cannot remove a unitig that has a node annotation")
        elif edit.operation == "add_edge":
            link_id = _added_link_id(proposal, edit)
            if any(link.link_id == link_id for link in cdbg.links):
                errors.append(f"edit {edit.edit_id} link id collides with an existing link")
        elif edit.operation == "mark_suspicious":
            namespace = str(edit.parameters.get("namespace", "qc"))
            feature = str(edit.parameters.get("feature", "suspicious"))
            target_type = str(edit.parameters.get("target_type", "node"))
            layer = _find_layer(cdbg, namespace, feature, target_type)
            if layer is not None and layer.dtype != "int64":
                errors.append(f"edit {edit.edit_id} incompatible annotation dtype: {layer.dtype}")
        elif edit.operation in {"change_node_annotation", "change_edge_annotation", "reassign_label"}:
            target_type = str(edit.parameters.get("target_type", "node" if edit.operation != "change_edge_annotation" else "edge"))
            if target_type == "internal_node":
                owner = next((row.unitig_id for row in cdbg.mapping if row.cfa_node_id == edit.target), None)
                if owner is not None and any(
                    other.operation in {"split_unitig", "merge_unitigs", "remove_node"}
                    and (other.target == owner or other.secondary_target == owner)
                    for other in proposal.edits
                ):
                    errors.append(f"contradictory edits use {edit.target}")
            if target_type == "internal_edge":
                for other in proposal.edits:
                    if other.operation != "split_unitig":
                        continue
                    unitig = unitigs.get(other.target)
                    cut = other.parameters.get("cut_after")
                    if (
                        unitig is not None
                        and isinstance(cut, int)
                        and 0 <= cut < len(unitig.internal_edge_ids)
                        and unitig.internal_edge_ids[cut] == edit.target
                    ):
                        errors.append(f"contradictory edits {other.edit_id} and {edit.edit_id} both use {edit.target}")
    return errors


def _target_errors(
    edit: GraphEdit,
    cdbg: Cdbg,
    unitigs: dict[str, Unitig],
    links: dict[str, Link],
    consumed_unitigs: dict[str, str],
    consumed_links: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    if edit.operation == "split_unitig":
        unitig = unitigs.get(edit.target)
        if unitig is None:
            return [f"nonexistent target: {edit.target}"]
        errors.extend(_split_errors(cdbg, unitig, edit))
    elif edit.operation == "merge_unitigs":
        errors.extend(_merge_errors(cdbg, edit, unitigs, links))
    elif edit.operation == "remove_node":
        if edit.target not in unitigs:
            return [f"nonexistent target: {edit.target}"]
        incident = [
            link.link_id
            for link in cdbg.links
            if link.source == edit.target or link.target == edit.target
        ]
        missing = [link_id for link_id in incident if consumed_links.get(link_id) is None]
        if missing:
            errors.append(
                f"remove_node {edit.target} leaves dangling links: {', '.join(missing)}"
            )
    elif edit.operation == "remove_edge":
        if edit.target not in links:
            internal = {
                edge_id
                for unitig in cdbg.unitigs
                for edge_id in unitig.internal_edge_ids
            }
            if edit.target in internal:
                errors.append(f"remove_edge cannot delete internal edge {edit.target}")
            else:
                errors.append(f"nonexistent target: {edit.target}")
    elif edit.operation == "add_edge":
        errors.extend(_add_edge_errors(cdbg, edit, unitigs, consumed_unitigs))
    elif edit.operation in {"change_node_annotation", "change_edge_annotation"}:
        errors.extend(_annotation_change_errors(cdbg, edit, unitigs, links, consumed_unitigs, consumed_links))
    elif edit.operation == "reassign_label":
        errors.extend(_reassign_errors(cdbg, edit, unitigs, links, consumed_unitigs, consumed_links))
    elif edit.operation == "mark_suspicious":
        errors.extend(_mark_errors(edit, unitigs, links, consumed_unitigs, consumed_links))
    return errors


def _split_errors(cdbg: Cdbg, unitig: Unitig, edit: GraphEdit) -> list[str]:
    errors: list[str] = []
    if "cut_base" in edit.parameters or "position" in edit.parameters:
        errors.append(f"edit {edit.edit_id} cuts inside a CFA node")
    cut = edit.parameters.get("cut_after")
    if isinstance(cut, bool) or not isinstance(cut, int):
        errors.append(f"edit {edit.edit_id} cut_after must be a member index")
        return errors
    last = len(unitig.members) - 1
    if cut < 0 or cut >= last:
        errors.append(f"edit {edit.edit_id} cut is outside the CFA member boundaries")
        return errors
    if _node_annotation_targets(cdbg, unitig.unitig_id):
        errors.append(f"edit {edit.edit_id} cannot split a unitig that has a node annotation")
    try:
        pieces, overlaps = _pieces(cdbg, unitig)
    except ContractError as exc:
        return errors + exc.errors
    left = _join(pieces[: cut + 1], overlaps[:cut])
    right = _join(pieces[cut + 1 :], overlaps[cut + 1 :])
    if _shorter_than_k(cdbg, left) or _shorter_than_k(cdbg, right):
        errors.append(f"edit {edit.edit_id} would make a unitig shorter than k")
    return errors


def _merge_errors(
    cdbg: Cdbg,
    edit: GraphEdit,
    unitigs: dict[str, Unitig],
    links: dict[str, Link],
) -> list[str]:
    if edit.target not in unitigs or edit.secondary_target not in unitigs:
        missing = edit.target if edit.target not in unitigs else edit.secondary_target
        return [f"nonexistent target: {missing}"]
    if edit.target == edit.secondary_target:
        return [f"edit {edit.edit_id} merges a unitig with itself"]
    link = _merge_link(cdbg, edit, unitigs, links, [])
    if link is None:
        return [f"edit {edit.edit_id} has no single forward link between the unitigs"]
    errors: list[str] = []
    if link.orientation not in (None, "++"):
        errors.append(f"edit {edit.edit_id} has invalid orientation {link.orientation!r}")
    outs = [item for item in cdbg.links if item.source == edit.target]
    ins = [item for item in cdbg.links if item.target == edit.secondary_target]
    if len(outs) != 1 or outs[0].link_id != link.link_id or len(ins) != 1 or ins[0].link_id != link.link_id:
        errors.append(f"edit {edit.edit_id} would hide a branch inside the merged unitig")
    overlap = _link_overlap(cdbg, link)
    if overlap is None:
        errors.append(f"edit {edit.edit_id} has no overlap")
    elif not _sequences_overlap(unitigs[edit.target].sequence, unitigs[edit.secondary_target].sequence, overlap):
        errors.append(f"edit {edit.edit_id} overlap mismatch")
    if _node_annotation_targets(cdbg, edit.target) or _node_annotation_targets(cdbg, edit.secondary_target or ""):
        errors.append(f"edit {edit.edit_id} cannot merge a unitig that has a node annotation")
    return errors


def _merge_link(
    cdbg: Cdbg,
    edit: GraphEdit,
    unitigs: dict[str, Unitig],
    links: dict[str, Link],
    errors: list[str],
) -> Link | None:
    if edit.target not in unitigs or not edit.secondary_target or edit.secondary_target not in unitigs:
        return None
    named = edit.parameters.get("link_id")
    if named is not None:
        link = links.get(named)
        if link is None:
            errors.append(f"nonexistent target: {named}")
            return None
        if link.source != edit.target or link.target != edit.secondary_target:
            errors.append(f"edit {edit.edit_id} link does not join the named unitigs")
            return None
        return link
    found = [
        link
        for link in cdbg.links
        if link.source == edit.target and link.target == edit.secondary_target
    ]
    if len(found) != 1:
        errors.append(f"edit {edit.edit_id} has no single forward link between the unitigs")
        return None
    return found[0]


def _add_edge_errors(
    cdbg: Cdbg,
    edit: GraphEdit,
    unitigs: dict[str, Unitig],
    consumed_unitigs: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    if edit.secondary_target is None:
        return [f"edit {edit.edit_id} add_edge requires a secondary target"]
    for endpoint in (edit.target, edit.secondary_target):
        if endpoint not in unitigs:
            errors.append(f"nonexistent target: {endpoint}")
        elif endpoint in consumed_unitigs:
            errors.append(f"contradictory edits {consumed_unitigs[endpoint]} and {edit.edit_id} both use {endpoint}")
    if errors:
        return errors
    orientation = edit.parameters.get("orientation", "++")
    if orientation not in ORIENTATIONS:
        errors.append(f"edit {edit.edit_id} has invalid orientation {orientation!r}")
    overlap = edit.parameters.get("overlap")
    graph_type = str(cdbg.metadata.get("graph_type", ""))
    if overlap is None and graph_type == "de_bruijn":
        errors.append(f"edit {edit.edit_id} has no overlap")
        return errors
    if overlap is None:
        return errors
    if isinstance(overlap, bool) or not isinstance(overlap, int) or overlap < 0:
        errors.append(f"edit {edit.edit_id} has invalid overlap")
        return errors
    if orientation not in (None, "++"):
        errors.append(f"edit {edit.edit_id} refuses a non-forward overlap check")
        return errors
    if not _sequences_overlap(unitigs[edit.target].sequence, unitigs[edit.secondary_target].sequence, overlap):
        errors.append(f"edit {edit.edit_id} overlap mismatch")
    return errors


def _annotation_change_errors(
    cdbg: Cdbg,
    edit: GraphEdit,
    unitigs: dict[str, Unitig],
    links: dict[str, Link],
    consumed_unitigs: dict[str, str],
    consumed_links: dict[str, str],
) -> list[str]:
    target_type = "node" if edit.operation == "change_node_annotation" else "edge"
    target_type = str(edit.parameters.get("target_type", target_type))
    errors = _known_annotation_target(edit, target_type, unitigs, links, cdbg, consumed_unitigs, consumed_links)
    namespace = edit.parameters.get("namespace")
    feature = edit.parameters.get("feature")
    dtype = edit.parameters.get("dtype")
    if not isinstance(namespace, str) or not isinstance(feature, str) or not isinstance(dtype, str):
        errors.append(f"edit {edit.edit_id} annotation namespace, feature, and dtype are required")
        return errors
    if "value" not in edit.parameters:
        errors.append(f"edit {edit.edit_id} annotation value is required")
    layer = _find_layer(cdbg, namespace, feature, target_type)
    if layer is not None and layer.dtype != dtype:
        errors.append(f"edit {edit.edit_id} incompatible annotation dtype: {dtype}")
    return errors


def _reassign_errors(
    cdbg: Cdbg,
    edit: GraphEdit,
    unitigs: dict[str, Unitig],
    links: dict[str, Link],
    consumed_unitigs: dict[str, str],
    consumed_links: dict[str, str],
) -> list[str]:
    target_type = str(edit.parameters.get("target_type", "node"))
    errors = _known_annotation_target(edit, target_type, unitigs, links, cdbg, consumed_unitigs, consumed_links)
    namespace = edit.parameters.get("namespace")
    feature = edit.parameters.get("feature")
    if not isinstance(namespace, str) or not isinstance(feature, str):
        errors.append(f"edit {edit.edit_id} label namespace and feature are required")
        return errors
    layer = _find_layer(cdbg, namespace, feature, target_type)
    if layer is None or layer.dtype != "category":
        errors.append(f"edit {edit.edit_id} has no categorical label layer to reassign")
        return errors
    if edit.target not in set(map(str, layer.target_ids.tolist())):
        errors.append(f"missing annotation target: {edit.target}")
    if "value" not in edit.parameters or not isinstance(edit.parameters.get("value"), str):
        errors.append(f"edit {edit.edit_id} label value must be a string")
    return errors


def _mark_errors(
    edit: GraphEdit,
    unitigs: dict[str, Unitig],
    links: dict[str, Link],
    consumed_unitigs: dict[str, str],
    consumed_links: dict[str, str],
) -> list[str]:
    target_type = str(edit.parameters.get("target_type", "node"))
    if target_type == "node" and edit.target not in unitigs:
        return [f"nonexistent target: {edit.target}"]
    if target_type == "edge" and edit.target not in links:
        return [f"nonexistent target: {edit.target}"]
    if target_type not in {"node", "edge"}:
        return [f"edit {edit.edit_id} mark_suspicious target_type must be node or edge"]
    owner = consumed_unitigs if target_type == "node" else consumed_links
    if edit.target in owner and owner[edit.target] != edit.edit_id:
        return [f"contradictory edits {owner[edit.target]} and {edit.edit_id} both use {edit.target}"]
    return []


def _known_annotation_target(
    edit: GraphEdit,
    target_type: str,
    unitigs: dict[str, Unitig],
    links: dict[str, Link],
    cdbg: Cdbg,
    consumed_unitigs: dict[str, str],
    consumed_links: dict[str, str],
) -> list[str]:
    if target_type == "node":
        known = edit.target in unitigs
        consumed = consumed_unitigs
    elif target_type == "edge":
        known = edit.target in links
        consumed = consumed_links
    elif target_type == "internal_node":
        known = edit.target in {row.cfa_node_id for row in cdbg.mapping}
        consumed = {}
    elif target_type == "internal_edge":
        known = edit.target in {edge for unitig in cdbg.unitigs for edge in unitig.internal_edge_ids}
        consumed = {}
    else:
        return [f"edit {edit.edit_id} has unknown target_type {target_type}"]
    if not known:
        return [f"nonexistent target: {edit.target}"]
    if edit.target in consumed and consumed[edit.target] != edit.edit_id:
        return [f"contradictory edits {consumed[edit.target]} and {edit.edit_id} both use {edit.target}"]
    return []


def _split_unitig(cdbg: Cdbg, proposal: EditProposal, edit: GraphEdit) -> None:
    unitig = _unitig(cdbg, edit.target)
    cut = int(edit.parameters["cut_after"])
    pieces, overlaps = _pieces(cdbg, unitig)
    left_id = _fresh_id(cdbg, f"u_{proposal.proposal_id}_{edit.edit_id}_a")
    right_id = _fresh_id(cdbg, f"u_{proposal.proposal_id}_{edit.edit_id}_b")
    left_members = unitig.members[: cut + 1]
    right_members = unitig.members[cut + 1 :]
    boundary = unitig.internal_edge_ids[cut]
    left = _child_unitig(
        cdbg,
        left_id,
        _join(pieces[: cut + 1], overlaps[:cut]),
        left_members,
        unitig.internal_edge_ids[:cut],
        unitig.internal_edge_colors[:cut],
        overlaps[:cut],
    )
    right = _child_unitig(
        cdbg,
        right_id,
        _join(pieces[cut + 1 :], overlaps[cut + 1 :]),
        right_members,
        unitig.internal_edge_ids[cut + 1 :],
        unitig.internal_edge_colors[cut + 1 :],
        overlaps[cut + 1 :],
    )
    cdbg.unitigs = [item for item in cdbg.unitigs if item.unitig_id != unitig.unitig_id] + [left, right]
    for link in cdbg.links:
        if link.target == unitig.unitig_id:
            link.target = left_id
        if link.source == unitig.unitig_id:
            link.source = right_id
    cdbg.links.append(
        Link(
            link_id=boundary,
            source=left_id,
            target=right_id,
            orientation="++",
            color_ids=list(unitig.internal_edge_colors[cut]),
            overlap=overlaps[cut],
        )
    )
    _assign_members(cdbg, left_id, left_members)
    _assign_members(cdbg, right_id, right_members)
    _move_annotation(cdbg, "internal_edge", "edge", boundary)
    provenance = cdbg.metadata["edit_provenance"]
    provenance["unitig_parents"][left_id] = [unitig.unitig_id]
    provenance["unitig_parents"][right_id] = [unitig.unitig_id]
    provenance["link_parents"][boundary] = boundary


def _merge_unitigs(cdbg: Cdbg, proposal: EditProposal, edit: GraphEdit) -> None:
    left = _unitig(cdbg, edit.target)
    right = _unitig(cdbg, edit.secondary_target or "")
    link = _require_merge_link(cdbg, edit)
    overlap = _link_overlap(cdbg, link)
    if overlap is None or not _sequences_overlap(left.sequence, right.sequence, overlap):
        raise ContractError([f"edit {edit.edit_id} overlap mismatch"])
    new_id = _fresh_id(cdbg, f"u_{proposal.proposal_id}_{edit.edit_id}_m")
    members = list(left.members) + list(right.members)
    sequence = left.sequence + right.sequence[overlap:]
    merged = _child_unitig(
        cdbg,
        new_id,
        sequence,
        members,
        list(left.internal_edge_ids) + [link.link_id] + list(right.internal_edge_ids),
        [list(colors) for colors in left.internal_edge_colors]
        + [list(link.color_ids)]
        + [list(colors) for colors in right.internal_edge_colors],
        junction_overlaps(left, cdbg.k) + [overlap] + junction_overlaps(right, cdbg.k),
    )
    cdbg.unitigs = [
        item for item in cdbg.unitigs if item.unitig_id not in {left.unitig_id, right.unitig_id}
    ] + [merged]
    kept: list[Link] = []
    for item in cdbg.links:
        if item.link_id == link.link_id:
            continue
        if item.target == left.unitig_id:
            item.target = new_id
        if item.source == right.unitig_id:
            item.source = new_id
        kept.append(item)
    cdbg.links = kept
    _assign_members(cdbg, new_id, members)
    _move_annotation(cdbg, "edge", "internal_edge", link.link_id)
    cdbg.metadata["edit_provenance"]["unitig_parents"][new_id] = [left.unitig_id, right.unitig_id]


def _remove_edge(cdbg: Cdbg, edit: GraphEdit) -> None:
    if any(edge_id == edit.target for unitig in cdbg.unitigs for edge_id in unitig.internal_edge_ids):
        raise ContractError([f"remove_edge cannot delete internal edge {edit.target}"])
    if not any(link.link_id == edit.target for link in cdbg.links):
        raise ContractError([f"nonexistent target: {edit.target}"])
    if _edge_annotation_targets(cdbg, edit.target):
        raise ContractError([f"edit {edit.edit_id} cannot remove a link that has an edge annotation"])
    cdbg.links = [link for link in cdbg.links if link.link_id != edit.target]


def _remove_node(cdbg: Cdbg, edit: GraphEdit) -> None:
    if _node_annotation_targets(cdbg, edit.target):
        raise ContractError([f"edit {edit.edit_id} cannot remove a unitig that has a node annotation"])
    members = set(_unitig(cdbg, edit.target).members)
    cdbg.unitigs = [unitig for unitig in cdbg.unitigs if unitig.unitig_id != edit.target]
    cdbg.mapping = [row for row in cdbg.mapping if row.cfa_node_id not in members]


def _add_edge(cdbg: Cdbg, proposal: EditProposal, edit: GraphEdit) -> None:
    orientation = edit.parameters.get("orientation", "++")
    overlap = edit.parameters.get("overlap")
    link_id = _added_link_id(proposal, edit)
    if any(link.link_id == link_id for link in cdbg.links):
        raise ContractError([f"edit {edit.edit_id} link id collides with an existing link"])
    cdbg.links.append(
        Link(
            link_id=link_id,
            source=edit.target,
            target=edit.secondary_target or "",
            orientation=orientation,
            color_ids=[],
            overlap=overlap,
        )
    )
    cdbg.metadata["edit_provenance"]["link_parents"][link_id] = None


def _apply_annotation_edit(cdbg: Cdbg, proposal: EditProposal, edit: GraphEdit) -> None:
    if edit.operation == "mark_suspicious":
        namespace = str(edit.parameters.get("namespace", "qc"))
        feature = str(edit.parameters.get("feature", "suspicious"))
        target_type = str(edit.parameters.get("target_type", "node"))
        _write_value(
            cdbg,
            proposal,
            edit,
            namespace=namespace,
            feature=feature,
            target_type=target_type,
            dtype="int64",
            value=1,
            create=True,
        )
        return
    if edit.operation == "reassign_label":
        _write_value(
            cdbg,
            proposal,
            edit,
            namespace=str(edit.parameters["namespace"]),
            feature=str(edit.parameters["feature"]),
            target_type=str(edit.parameters.get("target_type", "node")),
            dtype="category",
            value=edit.parameters["value"],
            create=False,
        )
        return
    target_type = "node" if edit.operation == "change_node_annotation" else "edge"
    _write_value(
        cdbg,
        proposal,
        edit,
        namespace=str(edit.parameters["namespace"]),
        feature=str(edit.parameters["feature"]),
        target_type=str(edit.parameters.get("target_type", target_type)),
        dtype=str(edit.parameters["dtype"]),
        value=edit.parameters["value"],
        create=True,
    )


def _write_value(
    cdbg: Cdbg,
    proposal: EditProposal,
    edit: GraphEdit,
    *,
    namespace: str,
    feature: str,
    target_type: str,
    dtype: str,
    value: Any,
    create: bool,
) -> None:
    layer = _find_layer(cdbg, namespace, feature, target_type)
    values: dict[str, Any] = {}
    if layer is not None:
        if layer.dtype != dtype:
            raise ContractError([f"edit {edit.edit_id} incompatible annotation dtype: {dtype}"])
        for target_id, stored in zip(layer.target_ids.tolist(), layer.values.tolist()):
            values[str(target_id)] = stored
        if not create and edit.target not in values:
            raise ContractError([f"missing annotation target: {edit.target}"])
    elif not create:
        raise ContractError([f"edit {edit.edit_id} has no categorical label layer to reassign"])
    values[edit.target] = value
    annotate_cdbg(
        cdbg,
        namespace=namespace,
        feature=feature,
        target_type=target_type,
        values=values,
        dtype=dtype,
        provenance=_edit_provenance(cdbg, proposal, edit),
        replace=layer is not None,
    )


def _edit_provenance(cdbg: Cdbg, proposal: EditProposal, edit: GraphEdit) -> dict[str, Any]:
    return {
        "source": proposal.source,
        "method": edit.operation,
        "version": edit.model_version,
        "parameters": {
            "edit_id": edit.edit_id,
            "reason": edit.reason,
            "model_id": edit.model_id,
            "confidence": edit.confidence,
        },
        "parent_graph_id": cdbg.metadata["graph_id"],
        "parent_schema_version": str(cdbg.metadata.get("schema_version", "")),
        "parent_contract_version": str(cdbg.metadata.get("contract_version", "")),
    }


def _retarget_annotation_graph_id(cdbg: Cdbg, graph_id: str) -> None:
    updated = []
    for layer in cdbg.annotations:
        parameters = dict(layer.provenance.parameters)
        parameters.setdefault("previous_parent_graph_id", layer.provenance.parent_graph_id)
        provenance = replace(layer.provenance, parent_graph_id=graph_id, parameters=parameters)
        updated.append(replace(layer, provenance=provenance))
    cdbg.annotations = updated


def _move_annotation(cdbg: Cdbg, source_type: str, dest_type: str, target_id: str) -> None:
    kept = []
    for layer in cdbg.annotations:
        if layer.target_type != source_type or target_id not in set(map(str, layer.target_ids.tolist())):
            kept.append(layer)
            continue
        index = [str(item) for item in layer.target_ids.tolist()].index(target_id)
        value = layer.values[index]
        mask = [position for position in range(layer.target_ids.shape[0]) if position != index]
        if mask:
            kept.append(
                replace(
                    layer,
                    target_ids=layer.target_ids[mask],
                    values=layer.values[mask],
                )
            )
        dest = _find_layer_in(kept, layer.namespace, layer.feature, dest_type)
        if dest is not None and (dest.dtype != layer.dtype or dest.kind != layer.kind):
            raise ContractError(
                [f"cannot move annotation {target_id}: destination dtype does not match"]
            )
        row_value = value.tolist() if hasattr(value, "tolist") else value
        values = {}
        if dest is not None:
            for existing_id, stored in zip(dest.target_ids.tolist(), dest.values.tolist()):
                values[str(existing_id)] = stored
        values[target_id] = row_value
        moved = annotate_cdbg(
            _layer_host(cdbg, kept),
            namespace=layer.namespace,
            feature=layer.feature,
            target_type=dest_type,
            values=values,
            dtype=layer.dtype,
            provenance={
                "source": layer.provenance.source,
                "method": layer.provenance.method,
                "version": layer.provenance.version,
                "parameters": dict(layer.provenance.parameters),
                "parent_graph_id": layer.provenance.parent_graph_id,
                "parent_schema_version": layer.provenance.parent_schema_version,
                "parent_contract_version": layer.provenance.parent_contract_version,
            },
            kind=layer.kind,
            replace=dest is not None,
        )
        kept = list(moved.annotations)
    cdbg.annotations = kept


def _layer_host(cdbg: Cdbg, layers: list) -> Cdbg:
    host = cdbg
    host.annotations = layers
    return host


def _child_unitig(
    cdbg: Cdbg,
    unitig_id: str,
    sequence: str,
    members: list[str],
    edge_ids: list[str],
    edge_colors: list[list[int]],
    overlaps: list[int],
) -> Unitig:
    colors: set[int] = set()
    by_node = {row.cfa_node_id: row for row in cdbg.mapping}
    for node_id in members:
        colors.update(by_node[node_id].color_ids)
    return Unitig(
        unitig_id=unitig_id,
        sequence=sequence,
        members=list(members),
        color_ids=sorted(colors),
        internal_edge_ids=list(edge_ids),
        internal_edge_colors=[list(group) for group in edge_colors],
        internal_overlaps=list(overlaps),
    )


def _assign_members(cdbg: Cdbg, unitig_id: str, members: list[str]) -> None:
    index = {row.cfa_node_id: row for row in cdbg.mapping}
    for ordinal, node_id in enumerate(members):
        row = index[node_id]
        row.unitig_id = unitig_id
        row.ordinal = ordinal


def _pieces(cdbg: Cdbg, unitig: Unitig) -> tuple[list[str], list[int]]:
    by_node = {row.cfa_node_id: row for row in cdbg.mapping}
    lengths = [by_node[node_id].length for node_id in unitig.members]
    overlaps = junction_overlaps(unitig, cdbg.k)
    return split_unitig(unitig.sequence, lengths, overlaps), overlaps


def _join(pieces: list[str], overlaps: list[int]) -> str:
    if not pieces:
        raise ContractError(["cannot join an empty unitig piece"])
    sequence = pieces[0]
    for piece, overlap in zip(pieces[1:], overlaps):
        if not _sequences_overlap(sequence, piece, overlap):
            raise ContractError(["overlap mismatch while joining unitig pieces"])
        sequence = sequence + piece[overlap:]
    return sequence


def _sequences_overlap(left: str, right: str, overlap: int) -> bool:
    if overlap < 0:
        return False
    if overlap == 0:
        return True
    if len(left) < overlap or len(right) < overlap:
        return False
    return left[-overlap:] == right[:overlap]


def _link_overlap(cdbg: Cdbg, link: Link) -> int | None:
    if link.overlap is not None:
        return link.overlap
    if str(cdbg.metadata.get("graph_type")) == "de_bruijn" and isinstance(cdbg.k, int):
        return cdbg.k - 1
    return None


def _shorter_than_k(cdbg: Cdbg, sequence: str) -> bool:
    return (
        str(cdbg.metadata.get("graph_type")) == "de_bruijn"
        and isinstance(cdbg.k, int)
        and len(sequence) < cdbg.k
    )


def _node_annotation_targets(cdbg: Cdbg, unitig_id: str) -> bool:
    return any(
        layer.target_type == "node" and unitig_id in set(map(str, layer.target_ids.tolist()))
        for layer in cdbg.annotations
    )


def _edge_annotation_targets(cdbg: Cdbg, link_id: str) -> bool:
    return any(
        layer.target_type == "edge" and link_id in set(map(str, layer.target_ids.tolist()))
        for layer in cdbg.annotations
    )


def _find_layer(cdbg: Cdbg, namespace: str, feature: str, target_type: str):
    return _find_layer_in(cdbg.annotations, namespace, feature, target_type)


def _find_layer_in(layers: list, namespace: str, feature: str, target_type: str):
    found = [
        layer
        for layer in layers
        if layer.namespace == namespace and layer.feature == feature and layer.target_type == target_type
    ]
    if not found:
        return None
    if len(found) > 1:
        raise ContractError([f"annotation layer already exists: {namespace}:{feature}:{target_type}"])
    return found[0]


def _unitig(cdbg: Cdbg, unitig_id: str) -> Unitig:
    for unitig in cdbg.unitigs:
        if unitig.unitig_id == unitig_id:
            return unitig
    raise ContractError([f"nonexistent target: {unitig_id}"])


def _require_merge_link(cdbg: Cdbg, edit: GraphEdit) -> Link:
    named = edit.parameters.get("link_id")
    found = [
        link
        for link in cdbg.links
        if link.source == edit.target and link.target == edit.secondary_target
        and (named is None or link.link_id == named)
    ]
    if len(found) != 1:
        raise ContractError([f"edit {edit.edit_id} has no single forward link between the unitigs"])
    return found[0]


def _fresh_id(cdbg: Cdbg, base: str) -> str:
    taken = {unitig.unitig_id for unitig in cdbg.unitigs}
    taken.update(link.link_id for link in cdbg.links)
    candidate = base
    suffix = 2
    while candidate in taken:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _added_link_id(proposal: EditProposal, edit: GraphEdit) -> str:
    return f"l_{proposal.proposal_id}_{edit.edit_id}"
