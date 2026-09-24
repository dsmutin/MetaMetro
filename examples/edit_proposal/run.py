"""Apply one split to the chain CDBG and print the child provenance."""

from metametro.edits import EditProposal, GraphEdit, apply_edit_proposal
from metametro.fixtures import chain_cdbg


def main() -> None:
    cdbg = chain_cdbg()
    parent = next(unitig for unitig in cdbg.unitigs if unitig.members[:1] == ["n000003"])
    proposal = EditProposal(
        proposal_id="split_chain",
        source="examples.edit_proposal",
        edits=(
            GraphEdit(
                edit_id="split-1",
                operation="split_unitig",
                target=parent.unitig_id,
                secondary_target=None,
                parameters={"cut_after": 0},
                reason="member boundary",
                confidence=1.0,
                model_id="example",
                model_version="0",
            ),
        ),
    )
    updated = apply_edit_proposal(proposal, cdbg)
    print(updated.metadata["graph_id"])
    for unitig_id, sources in updated.metadata["edit_provenance"]["unitig_parents"].items():
        print(unitig_id, sources)
    if cdbg.metadata["graph_id"] == updated.metadata["graph_id"]:
        raise SystemExit("input graph_id changed")


if __name__ == "__main__":
    main()
