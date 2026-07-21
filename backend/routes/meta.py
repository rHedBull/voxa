"""Voxa API routes: meta."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response

from app.constants import *  # noqa: F401,F403
from app.schemas import *  # noqa: F401,F403
from app import constants  # live LIDAR_ROOT
from app.core import *  # noqa: F401,F403
from app.core import _voxa_class_name_to_id

router = APIRouter()


@router.get("/api/health")
def health():
    return {
        "status": "ok",
        "data_dir": str(DATA_DIR),
        "lidar_root": str(constants.LIDAR_ROOT) if constants.LIDAR_ROOT else None,
    }

@router.get("/api/config", response_model=ConfigResponse)
def get_config():
    if not CONFIG_PATH.exists():
        # Sensible defaults if no yaml is present.
        # class_id must always be a real id — the frontend keys palettes and
        # instance rows by it unconditionally.
        return ConfigResponse(classes=[
            ClassDef(id="boss",     label="Boss",       color="#5b8def", hotkey="1", class_id=0),
            ClassDef(id="fastener", label="Fastener",   color="#f5a524", hotkey="2", class_id=1),
            ClassDef(id="gasket",   label="Gasket",     color="#10b981", hotkey="3", class_id=2),
            ClassDef(id="fitting",  label="Fitting",    color="#d4a017", hotkey="4", class_id=3),
            ClassDef(id="rail",     label="Rail",       color="#a855f7", hotkey="5", class_id=4),
            ClassDef(id="plate",    label="Base plate", color="#64748b", hotkey="6", class_id=5),
            ClassDef(id="unknown",  label="Unknown",    color="#ef4444", hotkey="0", class_id=6),
        ]) # TODO: defaults make no sense
    with CONFIG_PATH.open() as f:
        raw = yaml.safe_load(f) or {}
    name_to_id = _voxa_class_name_to_id()
    classes = []
    for i, (cid, body) in enumerate((raw.get("classes") or {}).items()):
        color = body.get("color", "#5b8def")
        if isinstance(color, list):
            r, g, b = (int(round(c * 255)) for c in color[:3])
            color = f"#{r:02x}{g:02x}{b:02x}"
        classes.append(ClassDef(
            id=cid,
            label=body.get("label", cid.title()),
            color=color,
            hotkey=str(body.get("key", body.get("hotkey", str(i + 1)))),
            class_id=name_to_id.get(str(cid).lower(), i),
            group=str(body.get("group", "")),
            frozen=bool(body.get("frozen", False)),
        ))
    return ConfigResponse(classes=classes)

@router.get("/api/scenes", response_model=list[SceneInfo])
def list_scenes():
    out: list[SceneInfo] = []
    for s in discover(DATA_DIR, constants.LIDAR_ROOT):
        annot_key = s.name if s.tier == "legacy" else s.scene_id.replace("/", "__")
        gt = (ANNOT_DIR / annot_key / "ground_truth.json").exists()
        pr = (ANNOT_DIR / annot_key / "predictions.json").exists() # TODO: is this still up to date?
        out.append(SceneInfo(
            id=s.scene_id,
            tier=s.tier,
            name=s.name,
            has_source=True,
            source_format=s.source_format,
            has_labels=s.has_labels,
            has_intensity=s.has_intensity,
            has_mesh=s.has_mesh,
            has_ground_truth=gt,
            has_predictions=pr,
            n_points=s.n_points,
        ))
    return out
# TODO: fix where to load the scenes from? adapted to new scene data saving schemaq v1.3? where a decimated point cloud is not a new scene