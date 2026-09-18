#!/usr/bin/env python3
"""Regression tests for nullable LlamaExtract request schemas; no API calls.

Run: python tests/test_llamaextract_nullability.py
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omni_extract_bench.harness.nullable_schema import restore_nullability
from omni_extract_bench.harness.providers import llamaextract


def adapt(schema):
    return restore_nullability(schema, llamaextract._adapt_schema(schema))


class LlamaExtractNullabilityTests(unittest.TestCase):
    def test_required_and_optional_fields_both_allow_null(self):
        schema = {"type": "object", "properties": {
            "amount": {"type": ["number", "null"], "description": "Amount if present"},
            "note": {"type": ["string", "null"]},
            "id": {"type": "string"}}, "required": ["amount", "id"]}
        result = adapt(schema)
        self.assertEqual(result["required"], ["amount", "id"])
        self.assertEqual(result["properties"]["amount"], {
            "description": "Amount if present",
            "anyOf": [{"type": "number"}, {"type": "null"}]})
        self.assertEqual(result["properties"]["note"], {
            "anyOf": [{"type": "string"}, {"type": "null"}]})
        self.assertEqual(result["properties"]["id"], {"type": "string"})

    def test_nullable_array_retains_complete_items_and_nested_nullability(self):
        schema = {"type": "object", "properties": {"rows": {
            "type": ["array", "null"], "minItems": 1, "description": "Rows if present",
            "items": {"type": "object", "properties": {
                "amount": {"type": ["number", "null"]}}, "required": ["amount"]}}},
            "required": ["rows"]}
        result = adapt(schema)
        rows = result["properties"]["rows"]
        self.assertEqual(rows["description"], "Rows if present")
        self.assertEqual(rows["anyOf"][1], {"type": "null"})
        branch = rows["anyOf"][0]
        self.assertEqual(branch["type"], "array")
        self.assertEqual(branch["minItems"], 1)
        self.assertEqual(branch["items"]["required"], ["amount"])
        self.assertEqual(branch["items"]["properties"]["amount"], {
            "anyOf": [{"type": "number"}, {"type": "null"}]})

    def test_nullable_object_preserves_properties_and_required(self):
        schema = {"type": "object", "properties": {"details": {
            "anyOf": [{"type": "object", "properties": {"name": {"type": "string"}},
                       "required": ["name"], "additionalProperties": False}, {"type": "null"}],
            "description": "Optional details"}}, "required": ["details"]}
        result = adapt(schema)
        self.assertEqual(result, schema)

    def test_nullable_enum_keeps_typed_nonnull_values_and_separate_null(self):
        for values, expected_type in [(["MILD", "HIGH", None], "string"),
                                      ([1, 2, None], "integer"),
                                      ([False, True, None], "boolean")]:
            with self.subTest(values=values):
                schema = {"type": "object", "properties": {"level": {"enum": values}},
                          "required": ["level"]}
                result = adapt(schema)
                self.assertEqual(result["properties"]["level"], {"anyOf": [
                    {"type": expected_type, "enum": values[:-1]}, {"type": "null"}]})
                self.assertEqual(result["required"], ["level"])

    def test_refs_to_nullable_types_are_restored(self):
        schema = {"type": "object", "$defs": {
            "Amount": {"type": ["number", "null"]},
            "Row": {"type": "object", "properties": {"amount": {"$ref": "#/$defs/Amount"}},
                    "required": ["amount"]}},
            "properties": {"rows": {"type": "array", "items": {"$ref": "#/$defs/Row"}}}}
        result = adapt(schema)
        self.assertNotIn("$ref", json.dumps(result))
        self.assertNotIn("$defs", result)
        item = result["properties"]["rows"]["items"]
        self.assertEqual(item["required"], ["amount"])
        self.assertEqual(item["properties"]["amount"], {
            "anyOf": [{"type": "number"}, {"type": "null"}]})

    def test_union_siblings_and_nullable_scalar_items_survive(self):
        for union in ("anyOf", "oneOf"):
            with self.subTest(union=union):
                schema = {"type": "object", "properties": {"values": {
                    union: [{"type": "array", "items": {"type": ["integer", "null"]}},
                            {"type": "null"}], "minItems": 2}}}
                result = adapt(schema)["properties"]["values"]
                self.assertEqual(result["anyOf"][0], {"type": "array", "minItems": 2,
                    "items": {"anyOf": [{"type": "integer"}, {"type": "null"}]}})

    def test_null_must_satisfy_all_constraints(self):
        # A null enum member alone is not sufficient if the declared type excludes it.
        cases = [{"type": "number", "enum": [1, None]},
                 {"type": ["number", "null"], "enum": [1]},
                 {"type": ["number", "null"], "const": 1},
                 {"type": ["number", "null"], "not": {"type": "null"}},
                 {"allOf": [{"type": ["number", "null"]}, {"type": "number"}]},
                 {"oneOf": [{"type": ["number", "null"]}, {"type": "null"}]}]
        for field in cases:
            with self.subTest(field=field):
                schema = {"type": "object", "properties": {"value": field}}
                self.assertEqual(adapt(schema), llamaextract._adapt_schema(schema))

    def test_field_names_matching_schema_keywords_are_ordinary_fields(self):
        schema = {"type": "object", "properties": {
            "properties": {"type": "object", "properties": {
                "required": {"type": ["boolean", "null"]}}, "required": ["required"]},
            "items": {"type": ["string", "null"]}}, "required": ["properties"]}
        result = adapt(schema)
        self.assertEqual(set(result["properties"]), {"properties", "items"})
        self.assertEqual(result["required"], ["properties"])
        inner = result["properties"]["properties"]
        self.assertEqual(set(inner["properties"]), {"required"})
        self.assertEqual(inner["required"], ["required"])
        self.assertEqual(inner["properties"]["required"], {
            "anyOf": [{"type": "boolean"}, {"type": "null"}]})

    def test_inputs_and_nonnullable_schemas_are_unchanged(self):
        schema = {"type": "object", "properties": {
            "amount": {"type": "number", "minimum": 0},
            "rows": {"type": "array", "items": {"type": "string"}}}, "required": ["amount"]}
        original = copy.deepcopy(schema)
        adapted = llamaextract._adapt_schema(schema)
        before = copy.deepcopy(adapted)
        result = restore_nullability(schema, adapted)
        self.assertEqual(result, adapted)
        self.assertEqual(schema, original)
        self.assertEqual(adapted, before)

    def test_unresolved_reference_does_not_grant_null(self):
        self.assertEqual(restore_nullability({"$ref": "#/$defs/Unknown"}, {"type": "number"}),
                         {"type": "number"})

    def test_request_contains_corrected_schema_and_response_is_not_rewritten(self):
        schema = {"type": "object", "properties": {"amount": {"type": ["number", "null"]}},
                  "required": ["amount"]}
        sent = []
        def request(method, url, key, headers=None, data=None):
            if method == "POST":
                sent.append(json.loads(data))
                return {"id": "job-1"}
            return {"id": "job-1", "status": "COMPLETED", "extract_result": {"amount": None}}
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            out = Path(tmp) / "result.json"
            pdf = Path(tmp) / "test.pdf"
            pdf.write_bytes(b"%PDF-1.4\n")
            with patch.object(llamaextract, "_project_id", return_value="project-1"), \
                 patch.object(llamaextract, "_upload", return_value="file-1"), \
                 patch.object(llamaextract, "_req", side_effect=request):
                llamaextract.run(pdf, schema, out, "test-key", 0)
            self.assertEqual(sent[0]["configuration"]["data_schema"], {
                "type": "object", "properties": {"amount": {
                    "anyOf": [{"type": "number"}, {"type": "null"}]}}, "required": ["amount"]})
            self.assertEqual(sent[0]["configuration"]["tier"], llamaextract.TIER)
            self.assertEqual(sent[0]["configuration"]["extraction_target"], "per_doc")
            self.assertEqual(json.loads(out.read_text())["result"], {"amount": None})


if __name__ == "__main__":
    unittest.main()
