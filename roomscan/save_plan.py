"""Step 7a: build the output JSON document and validate it against schema/plan.schema.json."""
import json
from pathlib import Path

import jsonschema

SCHEMA = json.loads((Path(__file__).resolve().parents[1] / "schema" / "plan.schema.json").read_text())


def build_plan(capture_id, tier, rooms, meta):
    warnings = [f"{r['id']}: {w}" for r in rooms for w in r.get("warnings", [])]
    return {"schema_version": "1.0", "capture": {"id": capture_id, "tier": tier},
            "rooms": rooms, "adjacency": [], "damage": [], "concealed_damage_flags": [],
            "scope_items": [], "warnings": warnings, "meta": meta}


def validate(plan):
    jsonschema.validate(plan, SCHEMA)
