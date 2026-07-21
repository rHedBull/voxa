# Primitive Vocabulary (Eval-Labeling Phase 0) — Design

**Date:** 2026-07-21
**Status:** Implemented (2026-07-21, branch `feat/primitive-vocab-phase0`; merge
guard added beyond plan after security review)
**Branch:** `feat/primitive-vocab-phase0`

## Problem

The eval-grade labeling spec
(`engine/research/fable/docs/superpowers/specs/2026-07-20-eval-labeling-design.md`,
"Tooling delta — Voxa", phase 0) requires labeling at the **primitive level**:
fine-grained classes organized in groups (pipe network, duct, electrical,
plant units, attachments, structure, catch-all, stuff), plus per-instance
metadata (`flags[]`, `subtype`, `insulated`, `note`) — **34 assignable
classes** in total (25 new + 9 reused). Voxa today has 15 broad classes
(v6), single-key hotkeys, and no instance metadata beyond label/class/confirmed.

Phase 0 is deliberately the smallest slice: it unblocks labeling in the new
vocabulary. Eval regions (phase 1), point categories + fragment components
(phase 2), and loader invariants/manifest in `scan-schema` (phase 3) are
separate efforts.

## Decisions settled with the user (2026-07-21)

- **Phase 0 only** in this effort.
- **Legacy broad classes are display-only from now on**: existing labels
  render, but no class in the frozen set can be newly assigned — in any
  session, old or new. All new labeling uses the primitive vocabulary.
- **Hotkeys become two-stroke chords by group** (group key → member key).
- **`unknown` (id 6) freezes with the rest** (strict reading of the
  eval-labeling spec's invariant 4). Consequence for the denoise feature is
  accepted: it stays wired to id 6, untouched, and is simply **not used**
  until phase 2 rewires it to the `artifact` point category. Its internal
  write path is exempt from the new backend guard and carries a phase-2 TODO.

## Non-goals

- **Canonical labeling-cloud import is split out as phase 0b** (decided
  2026-07-21). The parent spec's phase-0 bullet also includes scene import
  producing the canonical labeling cloud (dense areas capped at 5 mm spacing,
  sparse kept native, ~100M-point clouds possible) plus the session-scale
  performance check. That gets its own spec→plan cycle; this effort ships
  vocabulary + metadata + freeze on existing scan.ply clouds.
- No point-category axis, no eval regions, no invariants/manifest (phases 1–3).
- No viewport rendering of flags/insulated (metadata is inspector-only).
- No migration of existing GT — old labels keep their ids; `collapse()` back
  to broad classes stays derivable because broad ids are frozen, never
  renumbered.
- No per-session vocabulary profiles — the freeze is global.

## Design

### 1. Class map: `classes.json` v6 → v7 (append-only, id reuse)

Canonical map: `engine/data/lidar/classes.json` (append-only policy, currently
v6, ids 0–14). Existing ids whose semantics match a primitive are **reused**;
broad classes superseded by finer primitives are **frozen**:

| id | name | fate |
|---|---|---|
| 0 | `pipe` | frozen — superseded by pipe-network group |
| 1 | `tank` | reused (plant unit) |
| 2 | `equipment` | reused; description narrowed to "nameplate machine not otherwise enumerated (generator, compressor, chiller, AHU — specific type in subtype)" |
| 3 | `structural` | frozen — superseded by structure things + `structure-region` stuff |
| 4 | `double` | frozen (already deprecated) |
| 5 | `fitting` | frozen — superseded by elbow/tee/reducer/flange/valve/cap |
| 6 | `unknown` | frozen (denoise exception, see Decisions) |
| 7–9 | `wall`/`floor`/`ceiling` | reused (stuff) |
| 10–11 | `beam`/`pillar` | reused (structure things) |
| 12 | `stair` | reused; description clarified to the `stair-flight` primitive (name stays `stair` — names are load-bearing for existing loaders) |
| 13 | `cable` | frozen — superseded by cable-tray/conduit/cable-bundle |
| 14 | `other` | reused; description clarified to `other-thing` (real, distinct, confidently none of the above) |

**25 appended ids (15–39), version 7:**

| ids | group | names |
|---|---|---|
| 15–21 | pipe-network | `pipe-straight`, `elbow`, `tee`, `reducer`, `flange`, `valve`, `cap` |
| 22–24 | duct | `duct-straight`, `duct-fitting`, `diffuser` |
| 25–27 | electrical | `cable-tray`, `conduit`, `cable-bundle` |
| 28–32 | plant-units | `pump`, `motor`, `vessel`, `heat-exchanger`, `cabinet` |
| 33–34 | attachments | `support`, `instrument` |
| 35–36 | structure | `railing`, `door` |
| 37–39 | stuff | `structure-region`, `mep-region`, `clutter` |

Descriptions come from the eval-labeling spec's cut rules (e.g. elbow/tee cut
at direction/diameter changes; cable-bundle is the resolution floor —
individual cables never separated; `mep-region` = recognizable-but-unresolvable
MEP material; `clutter` = unnameable aggregate). Stuff classes are ordinary
classes in phase 0 — region semantics arrive with phases 1–2.

The updated `classes.json` is committed in `engine/data/lidar` (data repo);
voxa's `config/classes.yaml` is rewritten to mirror it. A backend test loads
both and asserts id/name agreement so they cannot drift; it **skips with an
explicit reason** when the lidar root is absent (CI / other checkouts), since
the canonical file lives outside the voxa repo.

### 2. Config format + `/api/config`

Each `classes.yaml` entry gains two optional fields:

- `group:` one of `pipe-network | duct | electrical | plant-units |
  attachments | structure | other | stuff | legacy`. The reused `other`
  (id 14, the parent spec's "catch-all thing" group) is the sole member of
  `other`. **Every frozen class carries `group: legacy`** — including
  `unknown` (6) and `double` (4), which are frozen-by-decision rather than
  superseded.
- `frozen: true` — display-only (render + counts + visibility toggles, never
  assignable).

`key:` becomes the **within-group** chord key. Group order and group chord
keys (`1`–`8`: pipe-network, duct, electrical, plant-units, attachments,
structure, other, stuff; legacy has no chord) are fixed in one frontend
constant (`label-tools.js` or a new `class-groups.js`), not per-entry config. `/api/config` passes `group`/`frozen`
through additively — Inspect/Compare and legacy readers see a superset and are
unaffected. Colors are assigned in per-group hue families (pipe-network greens,
duct teals, electrical yellows, plant-units blues/reds, attachments purples,
structure oranges, other white/neutral, stuff muted greys, legacy desaturated)
so the viewport stays legible at 34 active classes.

### 3. Picker UI + chord hotkeys

- **Left-rail class list** renders grouped under collapsible headers. The
  **Legacy** group is collapsed by default, entries greyed with a `frozen`
  affordance; their visibility toggles and per-class counts still work.
- **Chords:** first keystroke `1`–`8` selects a group and shows a transient
  overlay listing the group's members with their second keys; the second
  keystroke classifies through the exact same shared apply pipeline the
  current single-key hotkeys use (so rapid-preseg classify = two keystrokes).
  `Esc` (or an invalid second key) cancels the pending group. The chord state
  machine is a pure function with unit tests; `mode-label.jsx` consumes it.
- **Class-picker modal** (Ctrl+Enter path) renders the same groups; frozen
  classes are omitted entirely.

### 4. Instance metadata (`Cuboid` fields + inspector)

`Cuboid` (backend `app/schemas.py`) gains optional pass-through fields:

```python
flags: list[str] = []        # validated subset of {"boundary_uncertain", "incomplete"}
subtype: str | None = None   # free text (gate vs. ball valve, pump model)
insulated: bool | None = None
note: str | None = None
```

They ride `instances_gt.json` untouched by any pipeline logic (no effect on
apply/export/compare in phase 0). The Instances-panel edit affordance (✎)
gains four inputs: two flag checkboxes, an insulated checkbox, a subtype text
field, a note textarea. Saving goes through the existing annotation
autosave/put path (atomic `put_annotation`; the no-silent-empty-doc invariant
is untouched).

### 5. Enforcement (write-side freeze)

Frontend omission alone cannot guarantee "no new legacy labels", so the
backend validates at the class-carrying request models: any request that
assigns a `class_id` (`ApplyShapeRequest`, the reassign path,
`CenterlineApplyRequest`, and the instance-cut partition's inherited class —
the implementation plan verifies the actual entry points; SAM-confirm and
cut-confirm classify through these shared paths, they are not separate
endpoints) rejects a frozen `class_id` with **422** and a message naming the
class and pointing at the primitive replacement group. The frozen set is
derived from the config (`frozen: true`), not hardcoded. This implements the
eval-labeling spec's loader-invariant 4 ("no new GT contains ids 4 or 6") at
the write side, plus the broader phase-0 freeze. The denoise route's internal
id-6 materialize is deliberately outside the guard (see Decisions).

**Freeze × existing legacy instances:** operations that would *newly assign*
a frozen class to points are blocked even when the class comes from an
existing instance — cutting a legacy-class instance (cut auto-inherits the
source class) and re-applying a legacy-class box both hit the 422; the
context-menu items are additionally disabled client-side with a tooltip
("legacy class — re-label with a primitive first"). This is deliberate strict
behavior, not a bug: the escape hatch is re-labeling the instance with a
primitive class first.

### 6. Testing

- **Backend:** frozen-class 422 per endpoint; yaml↔json id/name-consistency
  test; `Cuboid` metadata round-trip through `instances_gt.json`; flags-subset
  validation.
- **Frontend:** chord state-machine unit tests (group select, classify,
  cancel, invalid key); grouped-picker rendering (jsdom); frozen-class
  exclusion from picker/modal/hotkeys.
- **Browser verification:** label one instance end-to-end via a chord
  (e.g. `1`,`2` → elbow), edit its subtype/flags in the inspector, reload,
  confirm persistence; screenshot the grouped picker and chord overlay.

## Out-of-repo change

`engine/data/lidar/classes.json` v7 append is part of this effort but lands in
the data repo, not voxa. The voxa consistency test pins the expectation.
