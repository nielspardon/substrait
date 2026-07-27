# SPDX-License-Identifier: Apache-2.0
"""Validator for the Substrait relation conformance corpus.

For every checked-in case in ``tests/relations/cases/*.json`` this:

1. Strictly parses the file into a ``substrait.test.RelationTestCase`` (rejecting
   unknown fields), proving it is well-formed against the current protos.
2. Independently recomputes the output schema from the plan with the in-repo
   schema-deriver and asserts it equals ``expected_schema`` -- making the
   derivation half correct by construction.
3. Structurally checks that every ``expected_rows`` row conforms to
   ``expected_schema`` (arity + per-column literal type/nullability). Row *value*
   correctness is intentionally out of scope for the spec repo -- it is validated
   post-release by real consumers.

Mirrors the strict-parse style of ``tests/test_proto_example_validator.py``.
"""

from pathlib import Path

import pytest
from google.protobuf import json_format

try:
    from substrait.test import relation_test_case_pb2 as tc
    from substrait import type_pb2
except ImportError as err:
    raise ImportError(
        "Protobuf bindings not found. Run 'buf generate' to generate them."
    ) from err

from tests.relations.deriver import derive_plan_schema

CASES_DIR = Path(__file__).parent / "relations" / "cases"

# Map each Expression.Literal oneof field to the Type oneof field it satisfies.
# Only scalar literals in current corpus scope are listed; extend as cases grow.
LITERAL_TO_TYPE_KIND = {
    "boolean": "bool",
    "i8": "i8",
    "i16": "i16",
    "i32": "i32",
    "i64": "i64",
    "fp32": "fp32",
    "fp64": "fp64",
    "string": "string",
    "binary": "binary",
    "date": "date",
}


def _case_files() -> list[Path]:
    return sorted(CASES_DIR.glob("*.json"))


def _load(path: Path) -> tc.RelationTestCase:
    """Strict-parse a case file, rejecting unknown fields."""
    message = tc.RelationTestCase()
    json_format.Parse(path.read_text(), message, ignore_unknown_fields=False)
    return message


# Parametrize by file so a failure names the offending case.
CASE_FILES = _case_files()
CASE_IDS = [p.stem for p in CASE_FILES]


def test_corpus_is_non_empty():
    """Guard against a silently empty corpus (e.g. builder wrote nowhere)."""
    assert CASE_FILES, f"no corpus cases found under {CASES_DIR}"


@pytest.mark.parametrize("path", CASE_FILES, ids=CASE_IDS)
def test_case_parses_strictly(path: Path):
    """Each case is well-formed protobuf-JSON with no unknown fields."""
    message = _load(path)
    assert message.name, f"{path.name}: missing case name"
    assert message.name == path.stem, (
        f"{path.name}: case name {message.name!r} does not match file stem"
    )


@pytest.mark.parametrize("path", CASE_FILES, ids=CASE_IDS)
def test_derived_schema_matches_expected(path: Path):
    """The independently derived schema equals the asserted expected_schema."""
    message = _load(path)
    derived = derive_plan_schema(message.plan)
    assert derived == message.expected_schema, (
        f"{path.name}: derived schema does not match expected_schema\n"
        f"derived:\n{derived}\nexpected:\n{message.expected_schema}"
    )


@pytest.mark.parametrize("path", CASE_FILES, ids=CASE_IDS)
def test_rows_conform_to_schema(path: Path):
    """Every expected row conforms structurally to expected_schema."""
    message = _load(path)
    column_types = list(message.expected_schema.struct.types)
    n = len(column_types)
    for r_idx, row in enumerate(message.expected_rows.rows):
        assert len(row.values) == n, (
            f"{path.name}: row {r_idx} has {len(row.values)} values, expected {n}"
        )
        for c_idx, (literal, col_type) in enumerate(
            zip(row.values, column_types, strict=True)
        ):
            _assert_literal_conforms(path, r_idx, c_idx, literal, col_type)


def _assert_literal_conforms(path, r_idx, c_idx, literal, col_type):
    """Assert a single literal conforms to a column type (kind + nullability)."""
    lit_kind = literal.WhichOneof("literal_type")
    assert lit_kind is not None, f"{path.name}: row {r_idx} col {c_idx}: empty literal"
    col_kind = col_type.WhichOneof("kind")

    # A typed-null literal only requires the kind carried on Type.null; it is
    # valid iff the column is nullable.
    if lit_kind == "null":
        assert _type_nullability(col_type) == type_pb2.Type.NULLABILITY_NULLABLE, (
            f"{path.name}: row {r_idx} col {c_idx}: null literal in a "
            f"non-nullable {col_kind} column"
        )
        assert literal.null.WhichOneof("kind") == col_kind, (
            f"{path.name}: row {r_idx} col {c_idx}: null literal typed "
            f"{literal.null.WhichOneof('kind')!r} in a {col_kind!r} column"
        )
        return

    expected_kind = LITERAL_TO_TYPE_KIND.get(lit_kind)
    assert expected_kind is not None, (
        f"{path.name}: row {r_idx} col {c_idx}: literal kind {lit_kind!r} "
        "not yet handled by the row-conformance check"
    )
    assert expected_kind == col_kind, (
        f"{path.name}: row {r_idx} col {c_idx}: literal kind {lit_kind!r} "
        f"does not match column type {col_kind!r}"
    )
    if literal.nullable:
        assert _type_nullability(col_type) == type_pb2.Type.NULLABILITY_NULLABLE, (
            f"{path.name}: row {r_idx} col {c_idx}: nullable literal in a "
            f"non-nullable {col_kind} column"
        )


def _type_nullability(col_type: "type_pb2.Type") -> int:
    """Read the nullability off whichever scalar kind a Type carries."""
    kind = col_type.WhichOneof("kind")
    return getattr(col_type, kind).nullability
