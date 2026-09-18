"""Restore nullable fields after LlamaExtract's non-null dialect conversion.

Keeping a field out of ``required`` allows it to be omitted; it does not allow an
explicit null. Preserve both choices independently when adapting a schema.
"""
import copy

from .dialects import resolve_refs


def _allows_null(schema):
    """Check null against the types, enums and combinators handled by the adapter."""
    if isinstance(schema, bool):
        return schema
    if not isinstance(schema, dict):
        return False
    if "$ref" in schema:
        # resolve_refs bounds recursive/deep schemas. Do not infer permission from
        # an unresolved reference just because it has no explicit type here.
        return False
    t = schema.get("type")
    if t is not None and "null" not in (t if isinstance(t, list) else [t]):
        return False
    if "enum" in schema and None not in schema["enum"]:
        return False
    if "const" in schema and schema["const"] is not None:
        return False
    if "anyOf" in schema and not any(_allows_null(s) for s in schema["anyOf"]):
        return False
    if "oneOf" in schema and sum(_allows_null(s) for s in schema["oneOf"]) != 1:
        return False
    if "allOf" in schema and not all(_allows_null(s) for s in schema["allOf"]):
        return False
    if "not" in schema and _allows_null(schema["not"]):
        return False
    return True


def _nonnull_branch(node):
    """Follow the same non-null branch and sibling merge as the provider adapter."""
    if not isinstance(node, dict):
        return {}
    for key in ("anyOf", "oneOf", "allOf"):
        branches = [x for x in node.get(key, []) if isinstance(x, dict)]
        chosen = next((x for x in branches if x.get("type") != "null"), None)
        if chosen is not None:
            merged = {k: v for k, v in node.items() if k not in ("anyOf", "oneOf", "allOf")}
            for k, v in chosen.items():
                merged.setdefault(k, v)
            return _nonnull_branch(merged)
    return node


def restore_nullability(original, adapted):
    """Restore null alternatives without mutating either input or changing required.

    Array items, object properties and enum constraints stay inside the complete
    non-null alternative. Walking schema positions, rather than arbitrary dicts,
    also preserves fields whose names happen to be JSON Schema keywords.
    """
    original = resolve_refs(original)

    def visit(old, new):
        result = copy.deepcopy(new)
        if not isinstance(old, dict) or not isinstance(new, dict):
            return result
        branch = _nonnull_branch(old)
        for key, child in new.get("properties", {}).items():
            original_child = branch.get("properties", {}).get(key)
            if original_child is not None:
                result["properties"][key] = visit(original_child, child)
        if isinstance(new.get("items"), dict) and isinstance(branch.get("items"), dict):
            result["items"] = visit(branch["items"], new["items"])
        if _allows_null(old) and not _allows_null(new):
            # A full non-null branch keeps array items and object properties together.
            description = result.pop("description", None)
            result = {"anyOf": [result, {"type": "null"}]}
            if description is not None:
                result["description"] = description
        return result

    return visit(original, adapted)
