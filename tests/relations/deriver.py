# SPDX-License-Identifier: Apache-2.0
"""In-repo, schema-only derivation for the relation conformance corpus.

Given a Substrait relation, this recomputes the output schema (field names +
types) independently of the ``expected_schema`` an author asserted. The corpus
validator compares the two, making the derivation half of every case correct by
construction.

Scope (deliberately narrow for this first round):

* Relations: ``ReadRel`` (base_schema), ``FilterRel``, ``SortRel``.
* ``RelCommon.Emit`` output remapping on any of the above.
* Flat, scalar schemas only.

This is NOT an execution engine -- it never touches row values. Relations and
schema shapes outside the scope above raise ``UnsupportedRelation`` /
``UnsupportedSchema`` rather than deriving something wrong silently; support is
added alongside the cases that need it.

Unset ``RelCommon.emit_kind`` means ``Direct``
==============================================
Leaving ``emit_kind`` unset -- and, equivalently, omitting the ``RelCommon``
section entirely -- is defined to mean ``Direct``: the relation outputs its
columns as is. This is the documented default (substrait-io/substrait#1142) and
the single most common relation shape in real plans, so the deriver treats an
unset ``emit_kind`` and an absent ``common`` as passthrough, identical to an
explicit ``Direct``.
"""

from substrait import algebra_pb2 as alg
from substrait import type_pb2 as type_pb2

# A derived column: its name and its type. NamedStruct.names is depth-first over
# nested fields, but for the flat scalar schemas in scope one column == one
# top-level (name, type) pair.
Column = tuple[str, type_pb2.Type]


class DerivationError(Exception):
    """Base class for derivation failures."""


class UnsupportedRelation(DerivationError):
    """A relation type the deriver does not yet handle."""


class UnsupportedSchema(DerivationError):
    """A schema shape (e.g. nested struct) the deriver does not yet handle."""


def _columns_from_named_struct(schema: type_pb2.NamedStruct) -> list[Column]:
    """Flatten a NamedStruct into (name, type) columns.

    Only flat scalar schemas are supported: a nested struct field would make the
    depth-first ``names`` list longer than ``struct.types`` and require recursive
    flattening, which no in-scope case needs yet.
    """
    types = list(schema.struct.types)
    names = list(schema.names)
    if len(types) != len(names):
        raise UnsupportedSchema(
            "nested schemas are not supported: "
            f"{len(names)} names but {len(types)} top-level types"
        )
    for t in types:
        if t.WhichOneof("kind") == "struct":
            raise UnsupportedSchema("nested struct fields are not supported")
    return list(zip(names, types, strict=True))


def _apply_emit(common: alg.RelCommon, columns: list[Column]) -> list[Column]:
    """Apply a relation's RelCommon.emit_kind to its input columns.

    ``Direct`` -- and an unset ``emit_kind`` (equivalently, an absent
    ``common``), which is defined to mean ``Direct`` -- passes columns through
    unchanged. ``Emit`` selects/reorders them by ``output_mapping`` indices.
    """
    if common.WhichOneof("emit_kind") != "emit":
        return columns
    out = []
    for idx in common.emit.output_mapping:
        if idx < 0 or idx >= len(columns):
            raise DerivationError(
                f"emit output_mapping index {idx} out of range "
                f"for {len(columns)} input columns"
            )
        out.append(columns[idx])
    return out


def _derive_columns(rel: alg.Rel) -> list[Column]:
    """Recompute the output columns of a relation."""
    kind = rel.WhichOneof("rel_type")
    if kind == "read":
        read = rel.read
        columns = _columns_from_named_struct(read.base_schema)
        return _apply_emit(read.common, columns)
    if kind == "filter":
        # FilterRel does not change the schema; it only drops rows.
        columns = _derive_columns(rel.filter.input)
        return _apply_emit(rel.filter.common, columns)
    if kind == "sort":
        # SortRel does not change the schema; it only reorders rows.
        columns = _derive_columns(rel.sort.input)
        return _apply_emit(rel.sort.common, columns)
    raise UnsupportedRelation(f"cannot derive schema for relation type {kind!r}")


def derive_schema(rel: alg.Rel) -> type_pb2.NamedStruct:
    """Derive the output NamedStruct of a relation.

    The struct itself is REQUIRED; per-field nullability is carried on each
    derived column type.
    """
    columns = _derive_columns(rel)
    return type_pb2.NamedStruct(
        names=[name for name, _ in columns],
        struct=type_pb2.Type.Struct(
            types=[t for _, t in columns],
            nullability=type_pb2.Type.NULLABILITY_REQUIRED,
        ),
    )


def derive_plan_schema(plan) -> type_pb2.NamedStruct:
    """Derive the output schema of a single-root Plan.

    The corpus uses one RelRoot per plan. RelRoot.names, when present, must match
    the derived names -- an extra invariant on top of type derivation.
    """
    roots = [pr for pr in plan.relations if pr.WhichOneof("rel_type") == "root"]
    if len(roots) != 1:
        raise DerivationError(
            f"expected exactly one RelRoot in the plan, found {len(roots)}"
        )
    root = roots[0].root
    schema = derive_schema(root.input)
    if list(root.names) and list(root.names) != list(schema.names):
        raise DerivationError(
            f"RelRoot.names {list(root.names)} do not match derived names "
            f"{list(schema.names)}"
        )
    return schema
