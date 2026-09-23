"""Feature registry for one CGT matrix.

Each entry describes one column of ``X_node`` or ``X_edge``. The entry is a
feature. It is not a colour, a training label, or a prediction. Column order
matches ``node_feature_names`` or ``edge_feature_names``. Those name lists
stay in metadata; the registry does not replace them.

A directory written before the registry omits the keys. Loading that
directory does not invent entries and does not change array values.
``normalization`` is recorded text. This module does not scale or center a
matrix.
"""

from __future__ import annotations

from typing import Any, Sequence

from metametro.errors import ContractError


FEATURE_TYPE = "feature"
REGISTRY_FIELDS = (
    "name",
    "feature_type",
    "namespace",
    "source_annotation",
    "normalization",
    "dtype",
    "positional",
)


def feature_registry_entry(
    name: str,
    *,
    namespace: str,
    source_annotation: str,
    positional: bool,
    normalization: str = "none",
    dtype: str = "float32",
) -> dict[str, Any]:
    """Return one feature-registry record.

    ``positional`` is true only for an unnamed ndarray. Those columns use
    names such as ``f0`` and an empty ``namespace``. An empty namespace is
    not a biological namespace. ``source_annotation`` is empty, or
    ``namespace:feature`` when the column was copied from a CDBG sidecar
    layer. ``normalization`` is stored as given. The default ``none`` means
    the converter did not transform the values.
    """
    if not isinstance(name, str) or name == "":
        raise ContractError(["feature registry name must be a non-empty string"])
    if not isinstance(namespace, str):
        raise ContractError([f"feature registry namespace for {name} must be a string"])
    if not isinstance(source_annotation, str):
        raise ContractError([f"feature registry source annotation for {name} must be a string"])
    if not isinstance(positional, bool):
        raise ContractError([f"feature registry positional flag for {name} must be a bool"])
    if positional and namespace != "":
        raise ContractError([f"positional feature {name} must not carry a namespace"])
    if not isinstance(normalization, str) or normalization == "":
        raise ContractError([f"feature registry normalization for {name} must be a non-empty string"])
    if dtype != "float32":
        raise ContractError([f"feature registry dtype for {name} must be float32"])
    return {
        "name": name,
        "feature_type": FEATURE_TYPE,
        "namespace": namespace,
        "source_annotation": source_annotation,
        "normalization": normalization,
        "dtype": dtype,
        "positional": positional,
    }


def build_feature_registry(
    names: Sequence[str],
    *,
    explicit_count: int,
    positional: bool,
    annotation_sources: Sequence[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Describe every column of one feature matrix.

    ``names[:explicit_count]`` are the caller's own columns. When
    ``positional`` is true the caller passed an ndarray and did not supply
    names, so those columns are marked positional and their namespace is
    empty. Later columns are sidecar selections.
    ``annotation_sources`` is ``(namespace, source_annotation)`` in column
    order. Normalization is ``none`` because the values are copied unchanged.
    """
    if isinstance(explicit_count, bool) or not isinstance(explicit_count, int) or explicit_count < 0:
        raise ContractError(["explicit feature count must be a non-negative integer"])
    if explicit_count + len(annotation_sources) != len(names):
        raise ContractError(["feature registry width does not match feature names"])
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, name in enumerate(names):
        if index < explicit_count:
            record = feature_registry_entry(
                str(name),
                namespace="",
                source_annotation="",
                positional=positional,
            )
        else:
            namespace, source_annotation = annotation_sources[index - explicit_count]
            record = feature_registry_entry(
                str(name),
                namespace=namespace,
                source_annotation=source_annotation,
                positional=False,
            )
        if record["name"] in seen:
            raise ContractError([f"duplicate feature name: {record['name']}"])
        seen.add(record["name"])
        records.append(record)
    return records


def validate_feature_registry(
    metadata: dict[str, Any],
    registry_key: str,
    names_key: str,
    width: int,
    errors: list[str],
) -> None:
    """Append registry errors. A missing registry is valid schema 1.0.

    This function does not read or write feature values. A recorded
    normalization is not applied.
    """
    if registry_key not in metadata:
        return
    records = metadata[registry_key]
    if not isinstance(records, list):
        errors.append(f"{registry_key} must be a list")
        return
    if len(records) != width:
        errors.append(f"{registry_key} does not match the feature width")
        return
    names = metadata.get(names_key)
    seen: set[str] = set()
    for index, record in enumerate(records):
        prefix = f"{registry_key}[{index}]"
        if not isinstance(record, dict):
            errors.append(f"{prefix} must be a mapping")
            continue
        missing = [field for field in REGISTRY_FIELDS if field not in record]
        if missing:
            errors.append(f"{prefix} missing {', '.join(missing)}")
            continue
        name = record["name"]
        if not isinstance(name, str) or name == "":
            errors.append(f"{prefix} has an empty name")
        elif name in seen:
            errors.append(f"{prefix} repeats feature name {name}")
        else:
            seen.add(name)
        if isinstance(names, list) and index < len(names) and name != names[index]:
            errors.append(f"{prefix} name does not match {names_key}")
        if record["feature_type"] != FEATURE_TYPE:
            errors.append(f"{prefix} feature_type must be feature")
        namespace = record["namespace"]
        if not isinstance(namespace, str):
            errors.append(f"{prefix} namespace must be a string")
        source = record["source_annotation"]
        if not isinstance(source, str):
            errors.append(f"{prefix} source_annotation must be a string")
        positional = record["positional"]
        if not isinstance(positional, bool):
            errors.append(f"{prefix} positional must be a bool")
        elif positional and namespace != "":
            errors.append(f"{prefix} positional feature must not carry a namespace")
        normalization = record["normalization"]
        if not isinstance(normalization, str) or normalization == "":
            errors.append(f"{prefix} normalization must be a non-empty string")
        if record["dtype"] != "float32":
            errors.append(f"{prefix} dtype must be float32")
