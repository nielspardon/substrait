# SPDX-License-Identifier: Apache-2.0
"""Deterministic builder for the Substrait relation conformance corpus.

Constructs ``substrait.test.RelationTestCase`` messages programmatically over the
generated protobuf bindings and emits one canonical protobuf-JSON file per case
into ``tests/relations/cases/``.

The output is deterministic (sorted keys, fixed indent, proto field names) so
that regenerating the corpus produces no diff -- CI relies on this via a
``git diff --exit-code`` guard, mirroring the ANTLR-parser drift guard.

Fixtures are intentionally *version-agnostic*: ``Plan.version`` is omitted from
the checked-in files and stamped with the real Substrait version at packaging
time.

Run with::

    pixi run generate-relation-tests
"""

import sys
from pathlib import Path

# Make the generated protobuf bindings importable when run standalone
# (`python tests/relations/build.py`). pytest gets this path from pyproject.toml.
_GEN_PYTHON = Path(__file__).resolve().parents[2] / "gen" / "proto" / "python"
if str(_GEN_PYTHON) not in sys.path:
    sys.path.insert(0, str(_GEN_PYTHON))

from google.protobuf import json_format  # noqa: E402

from substrait import algebra_pb2 as alg  # noqa: E402
from substrait import plan_pb2  # noqa: E402
from substrait import type_pb2  # noqa: E402
from substrait.test import relation_test_case_pb2 as tc  # noqa: E402

CASES_DIR = Path(__file__).parent / "cases"

# Shorthand for nullability enum values.
REQUIRED = type_pb2.Type.NULLABILITY_REQUIRED
NULLABLE = type_pb2.Type.NULLABILITY_NULLABLE  # used by cases with nullable columns


# --------------------------------------------------------------------------- #
# Type / schema helpers
# --------------------------------------------------------------------------- #
def i32(nullability=REQUIRED) -> type_pb2.Type:
    """A 32-bit integer type."""
    return type_pb2.Type(i32=type_pb2.Type.I32(nullability=nullability))


def string(nullability=REQUIRED) -> type_pb2.Type:
    """A string type."""
    return type_pb2.Type(string=type_pb2.Type.String(nullability=nullability))


def named_struct(fields: list[tuple[str, type_pb2.Type]]) -> type_pb2.NamedStruct:
    """Build a NamedStruct from (name, type) pairs.

    The struct itself is REQUIRED; individual field nullability rides on each
    field type.
    """
    return type_pb2.NamedStruct(
        names=[name for name, _ in fields],
        struct=type_pb2.Type.Struct(
            types=[t for _, t in fields],
            nullability=REQUIRED,
        ),
    )


# --------------------------------------------------------------------------- #
# RelCommon helpers
# --------------------------------------------------------------------------- #
def direct() -> alg.RelCommon:
    """A RelCommon whose emit_kind is an explicit ``Direct{}`` (passthrough).

    Every relation in the corpus sets ``emit_kind`` explicitly -- the corpus
    assumes the proposed rule that ``RelCommon.emit_kind`` must be one of
    ``Direct``/``Emit`` and that an unset ``emit_kind`` is not conformant. No
    case relies on the historical "unset == Direct" interpretation.
    """
    return alg.RelCommon(direct=alg.RelCommon.Direct())


# --------------------------------------------------------------------------- #
# Literal / row helpers
# --------------------------------------------------------------------------- #
def lit_i32(value: int) -> alg.Expression.Literal:
    """A required i32 literal."""
    return alg.Expression.Literal(i32=value)


def lit_string(value: str) -> alg.Expression.Literal:
    """A required string literal."""
    return alg.Expression.Literal(string=value)


def _literal_expr(literal: alg.Expression.Literal) -> alg.Expression:
    return alg.Expression(literal=literal)


def virtual_table(
    schema: type_pb2.NamedStruct,
    rows: list[list[alg.Expression.Literal]],
) -> alg.Rel:
    """A ReadRel over an inline VirtualTable of literal rows.

    Each row is a list of literals, one per column of ``schema``. The ReadRel
    carries an explicit ``Direct`` emit_kind so no case relies on an unset
    ``emit_kind`` (see ``direct()``).
    """
    return alg.Rel(
        read=alg.ReadRel(
            common=direct(),
            base_schema=schema,
            virtual_table=alg.ReadRel.VirtualTable(
                expressions=[
                    alg.Expression.Nested.Struct(fields=[_literal_expr(v) for v in row])
                    for row in rows
                ],
            ),
        )
    )


def row(*values: alg.Expression.Literal) -> tc.ExpectedData.Row:
    """An expected-output row from literals."""
    return tc.ExpectedData.Row(values=list(values))


# --------------------------------------------------------------------------- #
# Case assembly
# --------------------------------------------------------------------------- #
def make_plan(root_rel: alg.Rel, output_names: list[str]) -> plan_pb2.Plan:
    """Wrap a relation in a RelRoot/Plan.

    ``Plan.version`` is deliberately omitted (version-agnostic fixture).
    """
    return plan_pb2.Plan(
        relations=[
            plan_pb2.PlanRel(
                root=alg.RelRoot(input=root_rel, names=output_names),
            )
        ],
    )


def case(
    name: str,
    description: str,
    behaviors: list[str],
    plan: plan_pb2.Plan,
    expected_schema: type_pb2.NamedStruct,
    expected_rows: list[tc.ExpectedData.Row],
    order: int = tc.CompareConfig.ROW_ORDER_MULTISET,
) -> tc.RelationTestCase:
    """Assemble a RelationTestCase."""
    return tc.RelationTestCase(
        name=name,
        description=description,
        behaviors=behaviors,
        plan=plan,
        expected_schema=expected_schema,
        expected_rows=tc.ExpectedData(rows=expected_rows),
        compare=tc.CompareConfig(order=order),
    )


# --------------------------------------------------------------------------- #
# Bring-up cases
# --------------------------------------------------------------------------- #
def filter_passthrough() -> tc.RelationTestCase:
    """FilterRel(true) over a VirtualTable: output schema == input schema.

    Exercises the trivial identity derivation used to stand up the harness.
    """
    schema = named_struct([("a", i32()), ("b", string())])
    read = virtual_table(
        schema,
        rows=[
            [lit_i32(1), lit_string("x")],
            [lit_i32(2), lit_string("y")],
        ],
    )
    filt = alg.Rel(
        filter=alg.FilterRel(
            common=direct(),
            input=read,
            condition=alg.Expression(literal=alg.Expression.Literal(boolean=True)),
        )
    )
    return case(
        name="filter_passthrough",
        description=(
            "A FilterRel with an explicit Direct emit and a constant-true "
            "condition over a two-column VirtualTable derives a schema "
            "identical to its input."
        ),
        behaviors=["filter_passthrough"],
        plan=make_plan(filt, output_names=["a", "b"]),
        expected_schema=schema,
        expected_rows=[
            row(lit_i32(1), lit_string("x")),
            row(lit_i32(2), lit_string("y")),
        ],
    )


def emit_remap_reorder() -> tc.RelationTestCase:
    """RelCommon.Emit output_mapping [2, 0] reorders/subsets the output columns.

    Exercises emit/remap -- the highest-divergence derivation behavior -- on a
    ReadRel: input (a, b, c) becomes output (c, a).
    """
    input_schema = named_struct([("a", i32()), ("b", string()), ("c", i32())])
    read = alg.Rel(
        read=alg.ReadRel(
            common=alg.RelCommon(
                emit=alg.RelCommon.Emit(output_mapping=[2, 0]),
            ),
            base_schema=input_schema,
            virtual_table=alg.ReadRel.VirtualTable(
                expressions=[
                    alg.Expression.Nested.Struct(
                        fields=[
                            alg.Expression(literal=lit_i32(1)),
                            alg.Expression(literal=lit_string("x")),
                            alg.Expression(literal=lit_i32(10)),
                        ]
                    ),
                    alg.Expression.Nested.Struct(
                        fields=[
                            alg.Expression(literal=lit_i32(2)),
                            alg.Expression(literal=lit_string("y")),
                            alg.Expression(literal=lit_i32(20)),
                        ]
                    ),
                ],
            ),
        )
    )
    expected_schema = named_struct([("c", i32()), ("a", i32())])
    return case(
        name="emit_remap_reorder",
        description=(
            "A ReadRel whose RelCommon.Emit output_mapping is [2, 0] outputs "
            "columns (c, a) -- reordered and subset from the input (a, b, c)."
        ),
        behaviors=["emit_remap"],
        plan=make_plan(read, output_names=["c", "a"]),
        expected_schema=expected_schema,
        expected_rows=[
            row(lit_i32(10), lit_i32(1)),
            row(lit_i32(20), lit_i32(2)),
        ],
    )


def emit_direct_passthrough() -> tc.RelationTestCase:
    """An explicit RelCommon.Direct{} passes the schema through unchanged.

    Complements emit_remap_reorder by exercising the other arm of the emit_kind
    oneof: `common { direct {} }`, the canonical schema-preserving emit. Every
    case in the corpus sets emit_kind explicitly (see ``direct()``); this case
    pins down that an explicit Direct is the identity on the output schema.
    """
    schema = named_struct([("a", i32()), ("b", string())])
    read = alg.Rel(
        read=alg.ReadRel(
            common=alg.RelCommon(direct=alg.RelCommon.Direct()),
            base_schema=schema,
            virtual_table=alg.ReadRel.VirtualTable(
                expressions=[
                    alg.Expression.Nested.Struct(
                        fields=[
                            alg.Expression(literal=lit_i32(1)),
                            alg.Expression(literal=lit_string("x")),
                        ]
                    ),
                    alg.Expression.Nested.Struct(
                        fields=[
                            alg.Expression(literal=lit_i32(2)),
                            alg.Expression(literal=lit_string("y")),
                        ]
                    ),
                ],
            ),
        )
    )
    return case(
        name="emit_direct_passthrough",
        description=(
            "A ReadRel with an explicit RelCommon.Direct{} emit derives a schema "
            "identical to its base_schema."
        ),
        behaviors=["emit_direct"],
        plan=make_plan(read, output_names=["a", "b"]),
        expected_schema=schema,
        expected_rows=[
            row(lit_i32(1), lit_string("x")),
            row(lit_i32(2), lit_string("y")),
        ],
    )


# All cases in the corpus. Keep sorted by name for stable file listing.
CASES = [
    emit_direct_passthrough,
    emit_remap_reorder,
    filter_passthrough,
]


def _to_json(message: tc.RelationTestCase) -> str:
    """Serialize a case to canonical, deterministic protobuf-JSON."""
    return (
        json_format.MessageToJson(
            message,
            sort_keys=True,
            preserving_proto_field_name=True,
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    written = set()
    for builder in CASES:
        message = builder()
        path = CASES_DIR / f"{message.name}.json"
        path.write_text(_to_json(message))
        written.add(path.name)
        print(f"wrote {path.relative_to(Path.cwd())}")

    # Remove stale cases so the directory exactly reflects CASES.
    for existing in CASES_DIR.glob("*.json"):
        if existing.name not in written:
            existing.unlink()
            print(f"removed stale {existing.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
