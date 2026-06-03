"""sessions/<session_id>/ store (scan-schema v2).

Owns session CRUD, the immutable preseg/source pins frozen at creation, and
default-session resolution (last worked). Pure I/O over ScanLayout paths;
the in-memory SegmentSession is constructed by app.core, not here.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from labeling.segment_io import (atomic_write_json, filter_tiny_segments,
                                 load_session_aux, save_session_aux,
                                 utc_now_iso)
from preseg.preseg_store import load_preseg, read_preseg_meta
from scenes.scan_layout import ScanLayout


def _validate_session_id(session_id: str) -> None:
    """session ids are single path segments; anything else could escape
    sessions/ when joined into filesystem paths (defense in depth — the
    routes' URL normalization already 404s traversal attempts)."""
    if (not session_id or "/" in session_id or "\\" in session_id
            or session_id in (".", "..") or session_id.startswith(".")):
        raise ValueError(f"invalid session_id {session_id!r}")


class PinMismatch(Exception):
    """A session's frozen pin no longer matches what is on disk."""

    def __init__(self, diverged: str, detail: str) -> None:
        super().__init__(detail)
        self.diverged = diverged  # "preseg" | "source"


@dataclass(frozen=True)
class SessionInfo:
    session_id: str
    name: str
    preseg_id: Optional[str]
    created_at: Optional[str]
    saved_at: Optional[str]
    dirty: bool
    has_output: bool
    corrupt: bool = False




def create_session(layout: ScanLayout, *, name: str, preseg_id: Optional[str],
                   n_points: int, source_fp: str,
                   min_segment_points: int = 0) -> SessionInfo:
    """Seed working arrays from prelabel/<preseg_id>/ (or all -1 for blank)
    and freeze both fingerprint pins. The only moment a preseg is chosen.
    ``min_segment_points`` > 0 drops seeded segments smaller than the
    threshold to (-1, -1) — seed-time analogue of the loader's old
    tiny-segment filter (a resumed session must round-trip byte-identical,
    so filtering happens here, never on resume). Bad/missing preseg_id is
    rejected by load_preseg — no separate validation needed here."""
    if preseg_id is not None:
        class_ids, instance_ids = load_preseg(layout, preseg_id, n_points)
        if min_segment_points > 0:
            class_ids, instance_ids = filter_tiny_segments(
                class_ids, instance_ids, min_segment_points)
        preseg_fp = read_preseg_meta(layout, preseg_id)["fingerprint"]
    else:
        class_ids = np.full(n_points, -1, dtype=np.int8)
        instance_ids = np.full(n_points, -1, dtype=np.int32)
        preseg_fp = None
    # NB: dash-separated stamp on purpose — history dirs use %Y%m%d_%H%M%S
    # and prune_history's _TS_RE must never match a session id.
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    session_id = f"{ts}_{preseg_id or 'blank'}"
    sp = layout.session(session_id)
    # exist_ok=False makes the create atomic — a same-second double create
    # fails here instead of silently overwriting the first writer's files.
    sp.dir.mkdir(parents=True, exist_ok=False)
    created_at = utc_now_iso()
    aux = {
        "preseg_id": preseg_id,
        "preseg_fingerprint": preseg_fp,
        "source_fingerprint": source_fp,
        "hidden_inst_ids": [],
        "dirty": False,
        "name": name,
        "created_at": created_at,
    }
    written = save_session_aux(sp.dir, aux,
                               class_ids=class_ids, instance_ids=instance_ids)
    return SessionInfo(session_id=session_id, name=name, preseg_id=preseg_id,
                       created_at=created_at, saved_at=written["saved_at"],
                       dirty=False, has_output=False)


def list_sessions(layout: ScanLayout) -> list[SessionInfo]:
    root = layout.sessions_root
    if not root.is_dir():
        return []
    out: list[SessionInfo] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        sp = layout.session(d.name)
        aux = load_session_aux(sp.dir)
        if aux is None:
            out.append(SessionInfo(session_id=d.name, name=d.name,
                                   preseg_id=None, created_at=None,
                                   saved_at=None, dirty=False,
                                   has_output=sp.output_gt_class_ids.exists(),
                                   corrupt=True))
            continue
        out.append(SessionInfo(
            session_id=d.name,
            name=aux.get("name") or d.name,
            preseg_id=aux.get("preseg_id"),
            created_at=aux.get("created_at"),
            saved_at=aux.get("saved_at"),
            dirty=bool(aux.get("dirty", False)),
            has_output=sp.output_gt_class_ids.exists(),
        ))
    return out


def rename_session(layout: ScanLayout, session_id: str, name: str) -> None:
    _validate_session_id(session_id)
    sp = layout.session(session_id)
    aux = load_session_aux(sp.dir)
    if aux is None:
        raise FileNotFoundError(f"session {session_id}: no readable session.json")
    aux["name"] = name
    atomic_write_json(sp.session_json, aux)


def delete_session(layout: ScanLayout, session_id: str) -> None:
    _validate_session_id(session_id)
    sp = layout.session(session_id)
    if not sp.dir.is_dir():
        raise FileNotFoundError(f"session {session_id} not found")
    shutil.rmtree(sp.dir)


def last_worked(infos: list[SessionInfo]) -> Optional[str]:
    """session_id with max saved_at (created_at fallback) from an
    already-fetched list — pure function so callers never read
    session.json twice. saved_at means last persisted EDIT (stamped by
    autosave/save); merely opening a session does not reorder."""
    live = [i for i in infos if not i.corrupt]
    if not live:
        return None
    return max(live, key=lambda i: (i.saved_at or i.created_at or "")).session_id


def verify_pins(layout: ScanLayout, session_id: str, *, source_fp: str) -> dict:
    """Enforce the immutable pins before resume. String-compares the session
    pin against the preseg's DECLARED fingerprint (prelabel/<id>/meta.json)
    — no array load or hashing on the load path; re-registering is the only
    supported way a preseg changes. Returns the aux dict on success so the
    caller doesn't re-read it."""
    sp = layout.session(session_id)
    aux = load_session_aux(sp.dir)
    if aux is None:
        raise FileNotFoundError(f"session {session_id}: no readable session.json")
    pinned_src = aux.get("source_fingerprint")
    if pinned_src and pinned_src != source_fp:
        raise PinMismatch("source", (
            f"session {session_id} is pinned to source {pinned_src} "
            f"but the loaded cloud is {source_fp}"))
    preseg_id = aux.get("preseg_id")
    if preseg_id is not None:
        try:
            current_fp = read_preseg_meta(layout, preseg_id)["fingerprint"]
        except FileNotFoundError:
            raise PinMismatch("preseg", (
                f"session {session_id} is pinned to preseg '{preseg_id}' "
                f"which no longer exists"))
        if aux.get("preseg_fingerprint") != current_fp:
            raise PinMismatch("preseg", (
                f"session {session_id}: preseg '{preseg_id}' content changed "
                f"(pinned {aux.get('preseg_fingerprint')}, now {current_fp})"))
    return aux
