# Primitive Vocabulary (Phase 0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the eval-labeling primitive vocabulary in voxa — classes.json v7 (34 assignable classes in 8 chord groups), frozen legacy classes with a backend write-guard, two-stroke chord hotkeys, grouped picker UI, and per-instance metadata (`flags[]`, `subtype`, `insulated`, `note`).

**Architecture:** Config-driven: `classes.yaml` gains `group`/`frozen` per class, passed through `/api/config` additively. One backend guard helper (`reject_frozen_class`) at every class-carrying entry point. One pure frontend chord module (`class-chords.js`) consumed by the three existing hotkey sites (global handler, class-picker modal, rapid mode). Metadata fields are inert pass-through on `Cuboid`.

**Tech Stack:** FastAPI + Pydantic (backend), React 18 + Vite (frontend), pytest + vitest.

**Spec:** `docs/superpowers/specs/2026-07-21-primitive-vocabulary-phase0-design.md`

**Conventions:** Backend tests run `.venv/bin/pytest backend/tests/...` from the repo root. Frontend tests run `npm run test:frontend` from the repo root (or `npx vitest run --root frontend src/<file>` for one file). Backend has no autoreload — irrelevant here until the final browser-verification task, where the dev server must be (re)started fresh. Commit after every task.

---

### Task 1: Canonical class map v7 + voxa classes.yaml + consistency test

**Files:**
- Modify: `/home/hendrik/coding/engine/data/lidar/classes.json` (NOT a git repo — plain file edit; the test below pins it)
- Rewrite: `config/classes.yaml`
- Test: `backend/tests/test_class_config.py` (new)

- [ ] **Step 1: Write the failing consistency test**

```python
"""classes.yaml ↔ canonical classes.json consistency (phase 0, spec §1)."""
import json
import os
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
YAML_PATH = REPO_ROOT / "config" / "classes.yaml"
CANON_PATH = Path(os.environ.get(
    "VOXA_CANON_CLASSES",
    "/home/hendrik/coding/engine/data/lidar/classes.json"))

GROUPS = {"pipe-network", "duct", "electrical", "plant-units",
          "attachments", "structure", "other", "stuff", "legacy"}


def _yaml_classes():
    raw = yaml.safe_load(YAML_PATH.read_text())
    return raw["classes"]


def test_yaml_ids_unique_and_grouped():
    classes = _yaml_classes()
    ids = [b["id"] for b in classes.values()]
    assert len(ids) == len(set(ids)), "duplicate class ids in classes.yaml"
    for name, body in classes.items():
        assert body.get("group") in GROUPS, f"{name}: missing/unknown group"
        if body.get("group") == "legacy":
            assert body.get("frozen") is True, f"{name}: legacy must be frozen"
        else:
            assert not body.get("frozen"), f"{name}: frozen class outside legacy group"


def test_yaml_assignable_count_and_chord_keys():
    classes = _yaml_classes()
    assignable = {n: b for n, b in classes.items() if not b.get("frozen")}
    assert len(assignable) == 34
    # Within a group, chord second-keys must be unique.
    by_group = {}
    for name, body in assignable.items():
        by_group.setdefault(body["group"], []).append(str(body["key"]))
    for group, keys in by_group.items():
        assert len(keys) == len(set(keys)), f"duplicate chord keys in {group}"


@pytest.mark.skipif(not CANON_PATH.exists(),
                    reason="canonical classes.json absent (non-archive checkout)")
def test_yaml_matches_canonical_classes_json():
    canon = json.loads(CANON_PATH.read_text())
    assert canon["version"] == 7
    canon_by_id = {c["id"]: c["name"] for c in canon["classes"]}
    for name, body in _yaml_classes().items():
        assert body["id"] in canon_by_id, f"{name}: id {body['id']} not canonical"
        assert canon_by_id[body["id"]] == name, (
            f"id {body['id']}: yaml name {name!r} != canonical {canon_by_id[body['id']]!r}")
    # Every canonical id is represented in the yaml (nothing silently dropped).
    yaml_ids = {b["id"] for b in _yaml_classes().values()}
    assert yaml_ids == set(canon_by_id)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest backend/tests/test_class_config.py -v`
Expected: FAIL (`group` missing on current yaml entries; canonical version is 6).

- [ ] **Step 3: Append v7 to the canonical classes.json**

Edit `/home/hendrik/coding/engine/data/lidar/classes.json`: set `"version": 7`, update the three reused-description entries, and append ids 15–39. Reused-description edits (ids stay, names stay):

- id 2 `equipment`: "Nameplate machine not otherwise enumerated (generator, compressor, chiller, AHU — specific type goes in the instance subtype). One unit = one nameplate."
- id 12 `stair`: "Stair flight (the stair-flight primitive; name kept for loader back-compat)."
- id 14 `other`: "Catch-all thing (other-thing): real, distinct, resolvable unit, confidently none of the enumerated classes."

Appended entries (id, name, description — descriptions condensed from the eval-labeling spec's cut rules):

```json
{"id": 15, "name": "pipe-straight", "description": "Straight pipe run. Cut at every direction/diameter change and every flange face; a weld cuts only if visible. Continuously curved flexible elements (hose, flex conduit) are ONE instance — curvature is never a cut."},
{"id": 16, "name": "elbow", "description": "Pipe elbow (direction change)."},
{"id": 17, "name": "tee", "description": "Pipe tee/branch fitting."},
{"id": 18, "name": "reducer", "description": "Pipe reducer (diameter change)."},
{"id": 19, "name": "flange", "description": "Flange; every flange face is a cut point."},
{"id": 20, "name": "valve", "description": "In-line valve (belongs to the pipe network; gate/ball/etc. goes in subtype)."},
{"id": 21, "name": "cap", "description": "Pipe end cap / blind."},
{"id": 22, "name": "duct-straight", "description": "Straight duct run (same cut logic as pipe)."},
{"id": 23, "name": "duct-fitting", "description": "Duct elbow/tee/transition."},
{"id": 24, "name": "diffuser", "description": "Air diffuser / grille terminal."},
{"id": 25, "name": "cable-tray", "description": "Cable tray; along a run, cut at branches/junctions."},
{"id": 26, "name": "conduit", "description": "Electrical conduit run."},
{"id": 27, "name": "cable-bundle", "description": "Bundle of cables — the resolution floor; individual cables are never separated."},
{"id": 28, "name": "pump", "description": "Pump unit (one nameplate = one instance)."},
{"id": 29, "name": "motor", "description": "Electric motor unit."},
{"id": 30, "name": "vessel", "description": "Pressure vessel / process vessel (tank stays id 1)."},
{"id": 31, "name": "heat-exchanger", "description": "Heat exchanger unit."},
{"id": 32, "name": "cabinet", "description": "Electrical/control cabinet."},
{"id": 33, "name": "support", "description": "Support/bracket/hanger/clamp — exists because of what it touches (terminates at / wraps its host). Independent spanning members are beam/pillar."},
{"id": 34, "name": "instrument", "description": "Instrument/gauge/sensor/HMI mounted on a host; in-line instruments belong to the network."},
{"id": 35, "name": "railing", "description": "Railing/handrail run."},
{"id": 36, "name": "door", "description": "Door (windows/skylights are NOT things — they belong to their wall/ceiling stuff region)."},
{"id": 37, "name": "structure-region", "description": "STUFF: generic structure region with no instance decomposition (incl. scaffolding — temporary, interpenetrating, never per-tube instances)."},
{"id": 38, "name": "mep-region", "description": "STUFF: recognizable MEP material whose units cannot be resolved in this scan (fused racks, far-range bundles). Scan-resolvability state, not a scene property."},
{"id": 39, "name": "clutter", "description": "STUFF: real material that cannot be named — scrap piles, rubble, mixed debris. Confirmed GT, no instance metrics."}
```

- [ ] **Step 4: Rewrite `config/classes.yaml`**

Full replacement content (keep the header comment block, extend it with one line: `# group: chord group (spec 2026-07-21); frozen: display-only legacy — never assignable.`):

```yaml
classes:
  # ── pipe-network (chord group 1) ─────────────────────────────
  pipe-straight:   { id: 15, label: Pipe straight,   color: "#22c55e", key: "1", group: pipe-network }
  elbow:           { id: 16, label: Elbow,           color: "#16a34a", key: "2", group: pipe-network }
  tee:             { id: 17, label: Tee,             color: "#4ade80", key: "3", group: pipe-network }
  reducer:         { id: 18, label: Reducer,         color: "#86efac", key: "4", group: pipe-network }
  flange:          { id: 19, label: Flange,          color: "#15803d", key: "5", group: pipe-network }
  valve:           { id: 20, label: Valve,           color: "#34d399", key: "6", group: pipe-network }
  cap:             { id: 21, label: Cap,             color: "#a7f3d0", key: "7", group: pipe-network }
  # ── duct (chord group 2) ─────────────────────────────────────
  duct-straight:   { id: 22, label: Duct straight,   color: "#14b8a6", key: "1", group: duct }
  duct-fitting:    { id: 23, label: Duct fitting,    color: "#0d9488", key: "2", group: duct }
  diffuser:        { id: 24, label: Diffuser,        color: "#5eead4", key: "3", group: duct }
  # ── electrical (chord group 3) ───────────────────────────────
  cable-tray:      { id: 25, label: Cable tray,      color: "#facc15", key: "1", group: electrical }
  conduit:         { id: 26, label: Conduit,         color: "#eab308", key: "2", group: electrical }
  cable-bundle:    { id: 27, label: Cable bundle,    color: "#fde047", key: "3", group: electrical }
  # ── plant-units (chord group 4) ──────────────────────────────
  tank:            { id: 1,  label: Tank,            color: "#ef4444", key: "1", group: plant-units }
  equipment:       { id: 2,  label: Equipment,       color: "#5b8def", key: "2", group: plant-units }
  pump:            { id: 28, label: Pump,            color: "#dc2626", key: "3", group: plant-units }
  motor:           { id: 29, label: Motor,           color: "#f87171", key: "4", group: plant-units }
  vessel:          { id: 30, label: Vessel,          color: "#b91c1c", key: "5", group: plant-units }
  heat-exchanger:  { id: 31, label: Heat exchanger,  color: "#fb7185", key: "6", group: plant-units }
  cabinet:         { id: 32, label: Cabinet,         color: "#60a5fa", key: "7", group: plant-units }
  # ── attachments (chord group 5) ──────────────────────────────
  support:         { id: 33, label: Support,         color: "#a855f7", key: "1", group: attachments }
  instrument:      { id: 34, label: Instrument,      color: "#c084fc", key: "2", group: attachments }
  # ── structure (chord group 6) ────────────────────────────────
  beam:            { id: 10, label: Beam,            color: "#f5a524", key: "1", group: structure }
  pillar:          { id: 11, label: Pillar,          color: "#fb923c", key: "2", group: structure }
  stair:           { id: 12, label: Stair flight,    color: "#fdba74", key: "3", group: structure }
  railing:         { id: 35, label: Railing,         color: "#f97316", key: "4", group: structure }
  door:            { id: 36, label: Door,            color: "#c2410c", key: "5", group: structure }
  # ── other (chord group 7) ────────────────────────────────────
  other:           { id: 14, label: Other thing,     color: "#e2e8f0", key: "1", group: other }
  # ── stuff (chord group 8) ────────────────────────────────────
  wall:            { id: 7,  label: Wall,            color: "#64748b", key: "1", group: stuff }
  floor:           { id: 8,  label: Floor,           color: "#475569", key: "2", group: stuff }
  ceiling:         { id: 9,  label: Ceiling,         color: "#94a3b8", key: "3", group: stuff }
  structure-region: { id: 37, label: Structure region, color: "#6b7280", key: "4", group: stuff }
  mep-region:      { id: 38, label: MEP region,      color: "#71717a", key: "5", group: stuff }
  clutter:         { id: 39, label: Clutter,         color: "#57534e", key: "6", group: stuff }
  # ── legacy (frozen, display-only — no chord) ─────────────────
  pipe:            { id: 0,  label: Pipe (legacy),       color: "#4d7c5f", key: "", group: legacy, frozen: true }
  structural:      { id: 3,  label: Structural (legacy), color: "#8a6d3b", key: "", group: legacy, frozen: true }
  double:          { id: 4,  label: Double (archived),   color: "#6b7280", key: "", group: legacy, frozen: true }
  fitting:         { id: 5,  label: Fitting (legacy),    color: "#8a7a4a", key: "", group: legacy, frozen: true }
  unknown:         { id: 6,  label: Exclude / Review (legacy), color: "#6b7280", key: "", group: legacy, frozen: true }
  cable:           { id: 13, label: Cable (legacy),      color: "#8a833b", key: "", group: legacy, frozen: true }
```

Note the yaml flow-mapping style is new to this file — verify `yaml.safe_load` parses it (it does; it's standard YAML) but keep block style instead if you prefer matching the old file. IDs and fields are what matter.

- [ ] **Step 5: Run the test — all three pass**

Run: `.venv/bin/pytest backend/tests/test_class_config.py -v`
Expected: 3 PASS.

- [ ] **Step 6: Run the existing backend suite** (the yaml rewrite must not break `_voxa_class_name_to_id` consumers — e.g. denoise's `unknown` lookup, which still resolves since the `unknown:` entry remains)

Run: `.venv/bin/pytest backend/tests -q`
Expected: all pass. If a test asserts on old yaml contents (labels/colors/keys), update that test to the new vocabulary.

- [ ] **Step 7: Commit**

```bash
git add config/classes.yaml backend/tests/test_class_config.py
git commit -m "feat: primitive vocabulary v7 — classes.yaml + canonical consistency test"
```

---

### Task 2: `/api/config` passes `group` + `frozen` through

**Files:**
- Modify: `backend/app/schemas.py:115-123` (`ClassDef`)
- Modify: `backend/routes/meta.py:23-54` (`get_config`)
- Test: `backend/tests/test_class_config.py` (extend)

- [ ] **Step 1: Write the failing test** (append to `test_class_config.py`)

```python
def test_api_config_carries_group_and_frozen(client):
    r = client.get("/api/config")
    assert r.status_code == 200
    classes = r.json()["classes"]
    by_id = {c["id"]: c for c in classes}
    assert by_id["elbow"]["group"] == "pipe-network"
    assert by_id["elbow"]["frozen"] is False
    assert by_id["pipe"]["group"] == "legacy"
    assert by_id["pipe"]["frozen"] is True
```

Use the existing FastAPI test-client fixture from `backend/tests/conftest.py` (check its name — likely `client`; if config tests need the real `config/classes.yaml`, check how `VOXA_CONFIG`/`CONFIG_PATH` is set under the test env and point it at the repo file if the fixture defaults elsewhere).

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest backend/tests/test_class_config.py::test_api_config_carries_group_and_frozen -v`
Expected: FAIL (KeyError `group`).

- [ ] **Step 3: Implement**

`ClassDef` gains:

```python
    group: str = ""          # chord group (spec 2026-07-21); "" for defaults path
    frozen: bool = False     # display-only legacy — never assignable
```

`get_config` yaml branch adds to the `ClassDef(...)` call:

```python
            group=str(body.get("group", "")),
            frozen=bool(body.get("frozen", False)),
```

- [ ] **Step 4: Run test — passes; run backend suite**

Run: `.venv/bin/pytest backend/tests -q`
Expected: all pass (additive fields; Inspect/Compare readers unaffected).

- [ ] **Step 5: Commit**

```bash
git add backend/app/schemas.py backend/routes/meta.py backend/tests/test_class_config.py
git commit -m "feat: /api/config carries class group + frozen flags"
```

---### Task 3: Backend write-guard — frozen classes 422

**Files:**
- Modify: `backend/app/core.py` (add `frozen_class_ids()` + `reject_frozen_class()` near `_coerce_class_id`, line ~299)
- Modify: `backend/routes/segment.py` — `segment_apply` (`set_class` at :19 and `reassign` at :30), `apply-shape` (:219), `centerline-apply` (:239), `_cut_shape_core` instance branch (:297)
- Test: `backend/tests/test_frozen_guard.py` (new)

- [ ] **Step 1: Write the failing tests**

Model the fixture on an existing endpoint test that builds an annotated scene + session (see `backend/tests/test_export_labels.py::client_with_annotated_scene` or the segment tests' fixtures — reuse, don't reinvent). Cases:

```python
FROZEN_ID = 0   # legacy `pipe`
LIVE_ID = 16    # `elbow`

def test_apply_set_class_frozen_422(seg_client):
    r = seg_client.post("/api/segment/apply", json={
        "op": "set_class", "indices": _b64_idx([0, 1]),
        "payload": {"class_id": FROZEN_ID}})
    assert r.status_code == 422
    assert "frozen" in r.json()["detail"]

def test_apply_reassign_frozen_422(seg_client): ...       # op=reassign, target_class=FROZEN_ID
def test_apply_shape_frozen_422(seg_client): ...           # POST /api/segment/apply-shape, target_class=FROZEN_ID
def test_centerline_apply_frozen_422(seg_client): ...      # POST /api/segment/centerline-apply, target_class=FROZEN_ID
def test_live_class_still_applies(seg_client): ...         # same call with LIVE_ID → 200
def test_frozen_class_by_name_422(seg_client): ...         # target_class="pipe" (string coercion path)
```

Plus the cut-inherit case: create an instance whose points carry a frozen class (write the working array directly through the session fixture, or apply with LIVE_ID then monkeypatch — simplest: seed `seg.class_ids` with FROZEN_ID and instance id, then POST `/api/segment/cut-shape` with an `instance` source and assert 422 mentioning "re-label").

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/pytest backend/tests/test_frozen_guard.py -v`
Expected: FAIL (currently 200s).

- [ ] **Step 3: Implement**

In `app/core.py` (no caching — the yaml is tiny and interactive endpoints re-read config elsewhere already; this also keeps tests env-flexible):

```python
def frozen_class_ids() -> set[int]:
    """Class ids marked `frozen: true` in classes.yaml (display-only legacy)."""
    if not CONFIG_PATH.exists():
        return set()
    with CONFIG_PATH.open() as f:
        raw = yaml.safe_load(f) or {}
    return {int(b["id"]) for b in (raw.get("classes") or {}).values()
            if b.get("frozen") and "id" in b}


def reject_frozen_class(class_id, context="assign"):
    """422 on any attempt to newly assign a frozen (legacy) class.

    Deliberately NOT called by the denoise route's internal id-6 materialize —
    that path is exempt until phase 2 rewires it to the artifact category
    (spec 2026-07-21 §Decisions).
    """
    if class_id is None:
        return
    cid = int(class_id)
    if cid in frozen_class_ids():
        raise HTTPException(422, (
            f"class id {cid} is frozen (legacy, display-only) — "
            f"label with a primitive class instead"
            + ("; re-label the source instance with a primitive class first"
               if context == "cut" else "")))
```

Call sites in `routes/segment.py` (each right after the class id is resolved):

- `segment_apply` `set_class`: `cid = _coerce_class_id(req.payload["class_id"]); reject_frozen_class(cid)` then pass `cid`.
- `segment_apply` `reassign`: same for `target_class` (skip when `None`).
- `apply-shape` (:219) and `centerline-apply` (:239): `reject_frozen_class(target_class)` after coercion.
- `_cut_shape_core` instance branch (:297): `reject_frozen_class(src_class, context="cut")` before the reassign.

Careful: `segment_apply` wraps its body in `except (KeyError, ValueError) → 400`; `HTTPException` passes through untouched (it is not caught) — verify with the test.

- [ ] **Step 4: Run tests — pass; full backend suite**

Run: `.venv/bin/pytest backend/tests -q`
Expected: all pass. If denoise tests fail, the guard leaked into an exempt path — fix the guard placement, not the test.

- [ ] **Step 5: Commit**

```bash
git add backend/app/core.py backend/routes/segment.py backend/tests/test_frozen_guard.py
git commit -m "feat: 422 write-guard for frozen legacy classes on all assign paths"
```

---

### Task 4: `Cuboid` metadata fields

**Files:**
- Modify: `backend/app/schemas.py:83-104` (`Cuboid`)
- Test: `backend/tests/test_instance_metadata.py` (new)

- [ ] **Step 1: Write the failing test**

```python
"""Cuboid metadata fields (phase 0 spec §4): inert pass-through + validation."""
import pytest
from pydantic import ValidationError
from app.schemas import Cuboid


def _base(**kw):
    return Cuboid(id="i1", cls="elbow", **kw)


def test_metadata_defaults():
    c = _base()
    assert c.flags == [] and c.subtype is None
    assert c.insulated is None and c.note == ""


def test_metadata_roundtrip():
    c = _base(flags=["boundary_uncertain", "incomplete"],
              subtype="ball valve", insulated=True, note="jacket transition W side")
    d = c.model_dump()
    assert Cuboid(**d) == c


def test_unknown_flag_rejected():
    with pytest.raises(ValidationError):
        _base(flags=["completely_made_up"])
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest backend/tests/test_instance_metadata.py -v`
Expected: FAIL (unknown fields ignored / flags attr missing).

- [ ] **Step 3: Implement** — add to `Cuboid` after `seq`:

```python
    # Eval-labeling phase-0 metadata (spec 2026-07-21 §4). Inert pass-through:
    # rides instances_gt.json, no effect on apply/export/compare.
    flags: list[str] = []            # ⊆ {boundary_uncertain, incomplete}
    subtype: Optional[str] = None    # free text (gate vs. ball valve, pump model)
    insulated: Optional[bool] = None
    note: str = ""

    @field_validator("flags")
    @classmethod
    def _flags_known(cls, v):
        bad = set(v) - {"boundary_uncertain", "incomplete"}
        if bad:
            raise ValueError(f"unknown instance flags: {sorted(bad)}")
        return v
```

(`field_validator` is already imported in this file — see `CenterlinePath`.)

- [ ] **Step 4: Run tests + backend suite; Step 5: Commit**

```bash
git add backend/app/schemas.py backend/tests/test_instance_metadata.py
git commit -m "feat: Cuboid metadata fields — flags/subtype/insulated/note"
```

---

### Task 5: Chord module (pure) — `class-chords.js`

**Files:**
- Create: `frontend/src/class-chords.js`
- Test: `frontend/src/class-chords.test.js` (new, node env — no pragma needed)

- [ ] **Step 1: Write the failing tests**

```js
import { describe, it, expect } from 'vitest';
import { CLASS_GROUPS, assignable, groupMembers, chordStep } from './class-chords.js';

const CLASSES = [
  { id: 'elbow', label: 'Elbow', hotkey: '2', group: 'pipe-network', frozen: false },
  { id: 'pipe-straight', label: 'Pipe straight', hotkey: '1', group: 'pipe-network', frozen: false },
  { id: 'wall', label: 'Wall', hotkey: '1', group: 'stuff', frozen: false },
  { id: 'pipe', label: 'Pipe (legacy)', hotkey: '', group: 'legacy', frozen: true },
];

describe('class-chords', () => {
  it('has 8 chorded groups + un-chorded legacy', () => {
    expect(CLASS_GROUPS.filter((g) => g.key).map((g) => g.key))
      .toEqual(['1', '2', '3', '4', '5', '6', '7', '8']);
    expect(CLASS_GROUPS.find((g) => g.id === 'legacy').key).toBeNull();
  });

  it('assignable() excludes frozen', () => {
    expect(assignable(CLASSES).map((c) => c.id)).not.toContain('pipe');
  });

  it('first stroke: group key selects the group', () => {
    expect(chordStep(null, '1', CLASSES)).toEqual(
      { type: 'group', group: CLASS_GROUPS[0] });
  });

  it('first stroke: non-group key passes through', () => {
    expect(chordStep(null, 'f', CLASSES)).toEqual({ type: 'pass' });
  });

  it('second stroke: member key classifies within the pending group only', () => {
    const g = CLASS_GROUPS.find((x) => x.id === 'pipe-network');
    const r = chordStep(g, '2', CLASSES);
    expect(r.type).toBe('class');
    expect(r.cls.id).toBe('elbow');
    // '1' in stuff vs pipe-network must not collide:
    expect(chordStep(g, '1', CLASSES).cls.id).toBe('pipe-straight');
  });

  it('second stroke: Escape or invalid key cancels', () => {
    const g = CLASS_GROUPS.find((x) => x.id === 'pipe-network');
    expect(chordStep(g, 'Escape', CLASSES)).toEqual({ type: 'cancel' });
    expect(chordStep(g, 'z', CLASSES)).toEqual({ type: 'cancel' });
  });

  it('groupMembers returns assignable members of one group', () => {
    expect(groupMembers('pipe-network', CLASSES).length).toBe(2);
    expect(groupMembers('legacy', CLASSES)).toEqual([]);
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `npx vitest run --root frontend src/class-chords.test.js`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `frontend/src/class-chords.js`**

```js
// Two-stroke class chords (phase-0 spec §3). Pure module: group table +
// chord state transition. Consumers hold `pendingGroup` (null | group) and
// feed every keydown through chordStep.

export const CLASS_GROUPS = [
  { id: 'pipe-network', key: '1', label: 'Pipe network' },
  { id: 'duct',         key: '2', label: 'Duct' },
  { id: 'electrical',   key: '3', label: 'Electrical' },
  { id: 'plant-units',  key: '4', label: 'Plant units' },
  { id: 'attachments',  key: '5', label: 'Attachments' },
  { id: 'structure',    key: '6', label: 'Structure' },
  { id: 'other',        key: '7', label: 'Other' },
  { id: 'stuff',        key: '8', label: 'Stuff' },
  { id: 'legacy',       key: null, label: 'Legacy (frozen)' },
];

export const assignable = (classes) => classes.filter((c) => !c.frozen);

export const groupMembers = (groupId, classes) =>
  assignable(classes).filter((c) => c.group === groupId);

// chordStep(pendingGroup, key, classes) →
//   {type:'group', group} | {type:'class', cls} | {type:'cancel'} | {type:'pass'}
export function chordStep(pendingGroup, key, classes) {
  if (pendingGroup == null) {
    const group = CLASS_GROUPS.find((g) => g.key === key);
    return group ? { type: 'group', group } : { type: 'pass' };
  }
  if (key === 'Escape') return { type: 'cancel' };
  const cls = groupMembers(pendingGroup.id, classes)
    .find((c) => String(c.hotkey) === key);
  return cls ? { type: 'class', cls } : { type: 'cancel' };
}
```

- [ ] **Step 4: Run — pass. Step 5: Commit**

```bash
git add frontend/src/class-chords.js frontend/src/class-chords.test.js
git commit -m "feat: pure two-stroke class-chord module"
```

---

### Task 6: Wire chords into the three hotkey consumers

**Files:**
- Modify: `frontend/src/mode-label.jsx` (global handler ~:1226-1248; help text ~:784)
- Modify: `frontend/src/class-picker.jsx` (modal)
- Modify: `frontend/src/fast-label.jsx` (`FastLabelKeys` ~:53)
- Create: `frontend/src/chord-overlay.jsx` (transient member overlay)
- Test: extend `frontend/src/class-chords.test.js` only if new pure logic emerges; the JSX wiring is covered by browser verification (Task 9) — mode-label has no jsdom harness (see existing comment at `mode-label.jsx:~1695`).

- [ ] **Step 1: Create `chord-overlay.jsx`**

```jsx
// Transient overlay after the first chord stroke: lists the pending group's
// members with their second keys. Purely presentational.
import { groupMembers } from './class-chords.js';

export function ChordOverlay({ group, classes }) {
  if (!group) return null;
  return (
    <div className="chord-overlay">
      <div className="chord-title">{group.key} — {group.label}</div>
      {groupMembers(group.id, classes).map((c) => (
        <div key={c.id} className="chord-row">
          <span className="chord-key">{c.hotkey}</span>
          <span className="class-swatch" style={{ background: c.color }} />
          <span>{c.label}</span>
        </div>
      ))}
      <div className="chord-hint">Esc to cancel</div>
    </div>
  );
}
```

Add matching styles in `frontend/src/app.css` next to the existing `.class-picker-*` rules (fixed position near viewport top-center, dark card, one row per class — copy the `.class-picker-row` look).

- [ ] **Step 2: Rewire `mode-label.jsx` global handler**

Add state `const [pendingGroup, setPendingGroup] = useState(null);`. Replace the block at :1228-1248 (`const cls = classes.find(...)` through its `return`) with:

```jsx
      // Two-stroke class chord (class-chords.js). First stroke picks a group
      // (overlay shows members), second stroke classifies through the same
      // dispatch the old single-key hotkeys used.
      const step = chordStep(pendingGroup, e.key, classes);
      if (step.type === 'group') { e.preventDefault(); setPendingGroup(step.group); return; }
      if (step.type === 'cancel') { e.preventDefault(); setPendingGroup(null); return; }
      if (step.type === 'class') {
        e.preventDefault();
        setPendingGroup(null);
        const cls = step.cls;
        if ((activeTool === 'sam' || activeTool === 'presegment')
          && segState && segState.samSelection.size > 0) {
          confirmSamSelection(cls);
        } else if (segState && segState.selection.size > 0) {
          confirmSegmentSelection(cls);
        } else if (activeTool === 'box' && selBox) {
          applyBox(cls);
        } else {
          setActiveClass(cls.id);
        }
        return;
      }
      // step.type === 'pass' → fall through to the non-class hotkeys below.
```

Keep the walk-mode `wasdqe` guard above it (second-stroke keys are digits, but the group keys `1-8` don't collide with WASD). Note the existing `Escape` handling elsewhere in this handler: the chord `cancel` branch must run **before** any other Escape consumer while a group is pending — check handler order; if another Escape branch sits earlier, gate it with `!pendingGroup`. Add the chord's Escape/keys to the pendingGroup dependency array of the effect. Render `<ChordOverlay group={pendingGroup} classes={classes} />` inside the mode's root. Also reset `setPendingGroup(null)` on tool switch (add to the existing tool-change effect) so a stale pending group can't linger.

Update the shortcuts help (~:784): replace the per-class listing with the 8 group chords (`1 <second> … 8 <second>`).

- [ ] **Step 3: Rewire `ClassPickerModal`**

Same pattern inside the modal's keydown effect: local `pendingGroup` state, `chordStep(pendingGroup, e.key, classes)`; `class` → `onPick(cls)`, `group` → set pending + render the group's members highlighted (or simply filter the list to that group until Esc). Render grouped sections using `CLASS_GROUPS`/`groupMembers`; frozen classes are omitted entirely (pass `assignable(classes)` — but simplest is for the modal itself to filter, so call sites stay unchanged). Update the hint line: "Group key, then class key · Esc to cancel".

- [ ] **Step 4: Rewire `FastLabelKeys`** (rapid mode) identically: local pending state, chordStep, `onPickClass(cls)` on `class`. `FastLabelHUD` can render the pending group name for feedback (optional, one line).

- [ ] **Step 5: Run frontend suite** — `npm run test:frontend`. Expected: all pass (chords module tests + untouched suites; if `context-menu`/picker jsdom tests assert flat class lists, update them for grouping/filtering).

- [ ] **Step 6: Commit**

```bash
git add frontend/src/class-chords.js frontend/src/chord-overlay.jsx frontend/src/mode-label.jsx frontend/src/class-picker.jsx frontend/src/fast-label.jsx frontend/src/app.css
git commit -m "feat: two-stroke class chords in global hotkeys, picker modal, rapid mode"
```

---

### Task 7: Grouped left-rail class list + frozen filtering in edit panel

**Files:**
- Modify: `frontend/src/mode-label.jsx` — class list render (~:1370-1390), instance-edit class pills (~:1662-1674)

- [ ] **Step 1: Group the left-rail list**

Replace the flat `classes.map(...)` with iteration over `CLASS_GROUPS`: for each group with members (`classes.filter((c) => c.group === g.id)` — NOT `groupMembers`, legacy must render), a header row (`{g.key ? g.key + ' · ' : ''}{g.label}`) then the existing `.class-row` markup. The **legacy** group is collapsed by default behind a `const [showLegacy, setShowLegacy] = useState(false);` toggle header ("Legacy (frozen) ▸/▾"); its rows get an extra `frozen` class (greyed via css `opacity: .45`), `onClick` does **not** `setActiveClass` for frozen rows, and the hotkey cell shows `—`. Visibility eye + counts keep working for frozen rows (display-only, not gone). Classes with an unknown/absent `group` (defaults path) render in a trailing ungrouped section — don't crash on them.

- [ ] **Step 2: Filter the edit-panel class pills** (:1663 `classes.map(...)`) to `assignable(classes)` — re-classing an instance to a legacy class must be impossible. Import from `class-chords.js`.

- [ ] **Step 3: Run frontend suite; commit**

```bash
git add frontend/src/mode-label.jsx frontend/src/app.css
git commit -m "feat: grouped class rail with collapsed frozen legacy; assignable-only edit pills"
```

---

### Task 8: Instance inspector metadata inputs + cut gating

**Files:**
- Modify: `frontend/src/mode-label.jsx` — edit panel, after the Class row (~:1674)
- Modify: `frontend/src/cut-eligibility.js` + `frontend/src/cut-eligibility.test.js`

- [ ] **Step 1: Metadata inputs** — insert after the Class `ins-row`:

```jsx
                    <div className="ins-row">
                      <label>Flags</label>
                      <div className="ins-flags">
                        {['boundary_uncertain', 'incomplete'].map((f) => (
                          <label key={f} className="ins-check">
                            <input type="checkbox"
                              checked={(inst.flags || []).includes(f)}
                              disabled={inst.confirmed}
                              onChange={(e) => {
                                const cur = new Set(inst.flags || []);
                                e.target.checked ? cur.add(f) : cur.delete(f);
                                updateInstance(inst.id, { flags: [...cur] });
                              }} />
                            {f.replace('_', ' ')}
                          </label>
                        ))}
                        <label className="ins-check">
                          <input type="checkbox"
                            checked={inst.insulated === true}
                            disabled={inst.confirmed}
                            onChange={(e) => updateInstance(inst.id, { insulated: e.target.checked })} />
                          insulated
                        </label>
                      </div>
                    </div>
                    <div className="ins-row">
                      <label>Subtype</label>
                      <input className="ins-input" placeholder="e.g. ball valve"
                        value={inst.subtype || ''}
                        disabled={inst.confirmed}
                        onChange={(e) => updateInstance(inst.id, { subtype: e.target.value || null })} />
                    </div>
                    <div className="ins-row">
                      <label>Note</label>
                      <textarea className="ins-input ins-note" rows={2}
                        value={inst.note || ''}
                        disabled={inst.confirmed}
                        onChange={(e) => updateInstance(inst.id, { note: e.target.value })} />
                    </div>
```

`updateInstance` already persists through the annotation autosave path — no new plumbing. Add `.ins-flags/.ins-check/.ins-note` css.

- [ ] **Step 2: Cut gating for frozen-class instances.** In `cut-eligibility.js`, extend the instance-list branch: the caller now passes `classFrozen: boolean` (mode-label computes it from the right-clicked row's class via a `frozenClsIds` set derived from `classes`); when true return ineligible with reason `'legacy class — re-label with a primitive first'` (surfaces as the disabled-item tooltip). Add the two test cases (frozen → ineligible with that reason; unfrozen unchanged) to `cut-eligibility.test.js`, and thread the parameter through the call site in `mode-label.jsx`. The backend 422 (Task 3) remains the backstop.

- [ ] **Step 3: Run frontend suite; commit**

```bash
git add frontend/src/mode-label.jsx frontend/src/cut-eligibility.js frontend/src/cut-eligibility.test.js frontend/src/app.css
git commit -m "feat: instance metadata inspector; cut disabled on frozen-class instances"
```

---

### Task 9: Docs + full verification

**Files:**
- Modify: `CLAUDE.md` (Label-mode bullet: chords, frozen legacy, metadata fields; class-config bullet: group/frozen/v7)
- Modify: `docs/superpowers/specs/2026-07-21-primitive-vocabulary-phase0-design.md` (Status → Implemented)
- Browser verification (see `browser-verification` skill; memory: use a throwaway session, restart any stale :8765 backend)

- [ ] **Step 1: Update docs** — CLAUDE.md's class-config paragraph (`config/classes.yaml` entry fields now include `group`/`frozen`; hotkeys are two-stroke chords; legacy classes display-only, backend 422-guarded; `Cuboid` metadata fields listed). Keep it to the existing bullet style.

- [ ] **Step 2: Full suites**

Run: `.venv/bin/pytest backend/tests -q && npm run test:frontend`
Expected: all pass. Report the actual counts.

- [ ] **Step 3: Browser verification** (restart `npm run dev` first — backend has no autoreload):
  1. Open Label mode on `annotated/reg_u10_hfr_ut`, **create a throwaway session** (apply auto-saves to disk).
  2. Grouped class rail renders; Legacy collapsed; screenshot.
  3. Chord `1` → overlay lists pipe-network members; screenshot; `2` with a preseg selection → elbow instance created.
  4. Ctrl+Enter picker: grouped, no legacy entries.
  5. Edit the new instance: set subtype "test", flag `incomplete`; reload page; fields persisted.
  6. Console: zero errors; network: no failed requests (the 422 path is exercised by curl instead: `curl -s -X POST .../api/segment/apply -d '{"op":"set_class","indices":"...","payload":{"class_id":0}}'` → 422).
  7. Delete the throwaway session.

- [ ] **Step 4: Commit docs; final commit**

```bash
git add CLAUDE.md docs/
git commit -m "docs: primitive vocabulary phase 0 — chords, frozen legacy, metadata"
```

---

### Out of scope (tracked, not in this plan)

- Canonical labeling-cloud import + 100M-pt scale check → **phase 0b** (own spec).
- Eval regions, point categories, `point_component_ids`, scan-schema invariants → phases 1–3.
- Denoise rewire to `artifact` → phase 2 (route deliberately guard-exempt; TODO marker lives on `reject_frozen_class`'s docstring).
