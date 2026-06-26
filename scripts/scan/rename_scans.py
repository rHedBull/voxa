"""Rename scans + source_ids to the <scene>_<vendor>[_<density>] convention.

Dry-run by default; pass --apply to mutate. The lidar archive is NOT under git,
so --apply first writes a full per-scan-dir backup tarball. See
docs/superpowers/specs/2026-06-26-scan-naming-convention-design.md.
"""
from __future__ import annotations

import argparse
import json
import re
import tarfile
from pathlib import Path

from scan_schema import atomic_write_json, is_valid_scan_name, is_valid_source_id

# Deterministic, reviewed map (spec §4). scans: dir/scan_name; sources: source_id.
RENAME_MAP = {
    "scans": {
        "matterport_mechanical_room": "mechanical_room_matterport",
        "matterport_parkhouse": "parkhouse_matterport",
        "navvis_vlx3_water_treatment": "water_treatment_navvis",
        "smart_ais_clean": "smart_ais_navvis",
        "navvis_mlx": "generator_concrete_navvis",
        "construction_site": "construction_site_navvis",
        "factory_large": "factory_navvis",
        "munich_water_pump": "water_pump_navvis_500k",
        "munich_water_pump_3m": "water_pump_navvis_3m",
    },
    "sources": {
        "construction_site_sample_data": "construction_site_navvis",
        "factory_large": "factory_navvis",
        "navvis_mlx_sample_data": "generator_concrete_navvis",
        "navvis_vlx_3_data_water_treatment_facility": "water_treatment_navvis",
        "sample_data_vlx3_processindustry_smart_ais": "smart_ais_navvis",
        "matterport_mechanical_room": "mechanical_room_matterport",
        "matterport_parkhouse": "parkhouse_matterport",
    },
}

# Architecture: PLAN (pure, build {path: rewritten_obj}) -> RESIDUAL (field-scoped,
# whitelisting the deferred variant_id/labeling_variant namespace + free-text notes)
# -> only THEN mutate (backup, write, rename). The residual gate runs BEFORE any
# disk write, so a missed reference aborts cleanly instead of leaving the archive
# half-migrated. README + notes are reported for manual review, never auto-edited.

# Keys that legitimately retain an old-name-equal value after rename:
#   variant_id / labeling_variant — the deferred namespace (spec non-goal), MUST keep old value
#   notes — free text, hand-reviewed like README
_RETAINED_KEYS = {"variant_id", "labeling_variant"}
_FREETEXT_KEYS = {"notes"}


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def _sub_name(s, old, new):
    """Replace whole-name occurrences of old with new inside a provenance string,
    on token boundaries (so construction_site doesn't match construction_site_navvis)."""
    return re.sub(rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])", new, s)


def _plan_scan(scan_dir: Path, old: str, new: str, src_map: dict) -> dict:
    """Pure: return {path: rewritten_obj} for one scan. No disk writes."""
    planned: dict[Path, dict] = {}

    mp = scan_dir / "meta.json"
    m = _load(mp)
    m["scan_name"] = new
    fr = m.get("frame")
    if isinstance(fr, dict) and fr.get("canonical_id") == f"{old}#local":
        fr["canonical_id"] = f"{new}#local"
    d = m.get("derivation")
    if isinstance(d, dict):
        if d.get("scan_id") == old:
            d["scan_id"] = new
        _apply_source_map_to_derivation(d, src_map)   # variant_id left untouched
    planned[mp] = m

    vp = scan_dir / "variants.json"
    if vp.exists():
        v = _load(vp)
        if v.get("scan_id") == old:
            v["scan_id"] = new
        if v.get("canonical_id") == f"{old}#local":
            v["canonical_id"] = f"{new}#local"
        for var in v.get("variants", []):            # variant_id / labeling_variant untouched
            if isinstance(var.get("path"), str):
                var["path"] = _sub_name(var["path"], old, new)
            if var.get("root_source_id") in src_map:
                var["root_source_id"] = src_map[var["root_source_id"]]
            if isinstance(var.get("source"), str):
                var["source"] = _sub_name(var["source"], old, new)
        planned[vp] = v

    rroot = scan_dir / "renders"
    if rroot.is_dir():
        for run in sorted(rroot.iterdir()):
            rm = run / "meta.json"
            if rm.exists():
                j = _load(rm)
                fr = j.get("frame")
                if isinstance(fr, dict) and fr.get("canonical_id") == f"{old}#local":
                    fr["canonical_id"] = f"{new}#local"
                gf = j.get("generated_from")
                if isinstance(gf, dict):
                    if gf.get("scan_id") == old:
                        gf["scan_id"] = new
                    if isinstance(gf.get("source"), str):
                        gf["source"] = _sub_name(gf["source"], old, new)
                planned[rm] = j
            mf = run / "manifest.json"
            if mf.exists():
                j = _load(mf)
                if j.get("scene") == old:
                    j["scene"] = new
                planned[mf] = j

    sroot = scan_dir / "sessions"
    if sroot.is_dir():
        for sess in sorted(sroot.iterdir()):
            ig = sess / "instances_gt.json"
            if ig.exists():
                j = _load(ig)
                if j.get("scene") == f"annotated/{old}":
                    j["scene"] = f"annotated/{new}"
                planned[ig] = j

    mm = scan_dir / "source" / "mesh.meta.json"
    if mm.exists():
        j = _load(mm)
        if j.get("scene") == old:
            j["scene"] = new
        planned[mm] = j

    return planned


def _apply_source_map_to_derivation(d: dict, src_map: dict):
    root = d.get("root")
    if isinstance(root, dict) and root.get("source_id") in src_map:
        root["source_id"] = src_map[root["source_id"]]
    parent = d.get("parent")
    if isinstance(parent, dict) and parent.get("ref") in src_map:
        parent["ref"] = src_map[parent["ref"]]


def _plan_sources(root: Path, src_map: dict) -> dict:
    sp = root / "raw" / "sources.json"
    body = _load(sp)
    for e in body.get("sources", []):
        if e.get("source_id") in src_map:
            e["source_id"] = src_map[e["source_id"]]
    return {sp: body}


def _residual(view: dict, olds: set) -> tuple[list[str], list[str]]:
    """Recursively scan every string VALUE in the post-migration `view`
    ({path: obj}) for a token-boundary occurrence of any old name.
    Returns (fail_hits, review_hits): hits under _RETAINED_KEYS are ignored
    (expected); hits under _FREETEXT_KEYS are 'review' (reported, non-fatal);
    everything else is a 'fail' (a reference the migration missed)."""
    pats = {o: re.compile(rf"(?<![A-Za-z0-9_]){re.escape(o)}(?![A-Za-z0-9_])") for o in olds}
    fail, review = [], []

    def walk(obj, where, key=None):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, where, k)
        elif isinstance(obj, list):
            for v in obj:
                walk(v, where, key)
        elif isinstance(obj, str):
            if key in _RETAINED_KEYS:
                return
            for o, p in pats.items():
                if p.search(obj):
                    msg = f"{where} [{key}]: residual {o!r} in {obj!r}"
                    (review if key in _FREETEXT_KEYS else fail).append(msg)

    for path, obj in view.items():
        walk(obj, path)
    return fail, review


def _backup(root: Path, scan_olds, tar_path: Path, log: list[str]):
    log.append(f"  backup -> {tar_path}")
    with tarfile.open(tar_path, "w") as t:
        for old in scan_olds:
            t.add(root / "annotated" / old, arcname=f"annotated/{old}")
        t.add(root / "raw" / "sources.json", arcname="raw/sources.json")


def run_migration(root: Path, rename_map: dict, apply: bool,
                  backup_name: str = "rename_backup.tar") -> dict:
    root = Path(root)
    scan_map, src_map = rename_map["scans"], rename_map["sources"]
    olds = set(scan_map) | set(src_map)
    log: list[str] = []

    # Preconditions (by OLD name; nothing mutated yet). Explicit raises, not
    # asserts — these guard against renaming a missing scan or colliding onto an
    # existing dir and must NOT be stripped under `python -O`.
    for old, new in scan_map.items():
        if not (root / "annotated" / old).is_dir():
            raise SystemExit(f"precondition: missing scan {old}")
        if (root / "annotated" / new).exists():
            raise SystemExit(f"precondition: collision, {new} already exists")
        if not is_valid_scan_name(new):
            raise SystemExit(f"precondition: target scan_name not valid: {new}")
    for new in src_map.values():
        if not is_valid_source_id(new):
            raise SystemExit(f"precondition: target source_id not valid: {new}")

    # PLAN — build the full post-migration view (pure).
    planned: dict[Path, dict] = {}
    for old, new in scan_map.items():
        planned.update(_plan_scan(root / "annotated" / old, old, new, src_map))
    planned.update(_plan_sources(root, src_map))

    # RESIDUAL (before mutating): planned objects + every other *.json (read from
    # disk, .bak* skipped) so an un-rewritten reference is caught, not written over.
    view = dict(planned)
    for jf in (root / "annotated").rglob("*.json"):
        if ".bak" in jf.name or jf in planned:
            continue
        view[jf] = _load(jf)
    fail, review = _residual(view, olds)
    for r in review:
        log.append(f"  REVIEW (free-text, not auto-edited): {r}")
    for old in scan_map:
        rd = root / "annotated" / old / "README.md"
        if rd.exists():
            log.append(f"  REVIEW (free text, not auto-edited): {rd}")
    if fail:
        raise SystemExit(
            "RESIDUAL old names in un-rewritten id-fields (fix the rewriter):\n"
            + "\n".join(fail))

    for p in planned:
        log.append(f"  rewrite {p}")

    # MUTATE — only now. Backup first (archive is not under git).
    if apply:
        _backup(root, scan_map.keys(), root / backup_name, log)
        for p, obj in planned.items():
            atomic_write_json(p, obj)
        for old, new in scan_map.items():
            (root / "annotated" / old).rename(root / "annotated" / new)

    print("\n".join(log))
    return {"log": log, "planned": [str(p) for p in planned], "review": review}


def main(argv=None):
    from datetime import datetime, timezone
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_migration(a.root, RENAME_MAP, apply=a.apply,
                  backup_name=f"rename_backup_{stamp}.tar")
    print(f"\n{'APPLIED' if a.apply else 'DRY-RUN (no changes)'}")


if __name__ == "__main__":
    main()
