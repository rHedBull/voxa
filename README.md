# Voxa — 3D scan studio

A unified viewer + labeler for industrial LiDAR / 3D point clouds. Replaces the
older `3d-labeler` and `industrial-point-labeler` tools with a single app
organized around three clearly separated modes.

| Mode | Purpose |
|------|---------|
| **Inspect** | Fast scrubby review of a scan — color-by toggles, point size, scene stats, camera presets |
| **Label** | Cuboid annotation with a class palette, per-instance inspector, hide-by-class, optional auto-fit |
| **Compare** | Side-by-side ground-truth vs prediction with synced cameras and a server-computed diff (precision / recall / F1 / IoU + per-instance TP/FP/FN table) |

## Quick start

```bash
# 1. Drop a scan into the data dir
./scripts/import_scene.sh water_pump ../3d-labeler/data/real/water_pump_8k/source.glb

# 2. Launch (creates a .venv on first run, ~30s)
./scripts/run.sh

# 3. Open http://127.0.0.1:8765
```

The first run creates `voxa/.venv` and installs `requirements.txt`.

## Layout

```
voxa/
├── backend/          FastAPI (loads PLY/GLB, persists JSON annotations, computes diff)
│   ├── main.py
│   ├── point_cloud.py    (PLY/GLB loader from 3d-labeler)
│   └── requirements.txt
├── frontend/         React + Three.js via CDN — no build step
│   ├── index.html
│   ├── app.css       (Linear/Figma-ish dark/light theme)
│   ├── app.jsx       (shell, scene picker, save protocol)
│   ├── viewer.jsx    (Three.js viewport, orbit, cuboid overlay)
│   ├── api.jsx       (thin client for /api/*)
│   ├── mode-inspect.jsx
│   ├── mode-label.jsx
│   ├── mode-compare.jsx
│   └── tweaks-panel.jsx  (theme + mode tweaks panel)
├── config/
│   └── classes.yaml      (label classes — id, color, hotkey)
├── data/
│   ├── scenes/<name>/source.{ply,glb}   (input)
│   └── annotations/<name>/{ground_truth,predictions}.json
└── scripts/
    ├── run.sh            (start server)
    └── import_scene.sh   (copy a PLY/GLB into data/scenes/)
```

## Data format

**Scene** — `data/scenes/<name>/source.ply` or `source.glb`.

**Annotation JSON** — `data/annotations/<name>/{ground_truth,predictions}.json`:

```json
{
  "scene": "water_pump",
  "kind": "gt",
  "instances": [
    {
      "id": "inst-ab12cd34",
      "cls": "pipe",
      "label": "Pipe 3",
      "color": "#22c55e",
      "center": [0.12, 0.40, -0.05],
      "size":   [0.20, 0.05, 0.05],
      "rotation": [0, 0, 0],
      "conf": 1.0,
      "source": "manual"
    }
  ],
  "meta": {}
}
```

`predictions.json` uses the same shape — drop a model's output there to get the
Compare-mode diff for free.

## Hotkeys

| Mode    | Key            | Action |
|---------|----------------|--------|
| Any     | `⌘S` / `Ctrl+S`| Save GT annotations |
| Label   | `A`            | Add a cuboid for the active class |
| Label   | `0`–`9`        | Set active class (and reassign selected instance) |
| Label   | `F`            | Frame the selected instance |
| Label   | `⌫` / `Del`    | Delete the selected instance |
| Viewer  | drag           | Orbit |
| Viewer  | `Shift`+drag   | Pan |
| Viewer  | scroll         | Zoom |

## Customizing classes

Edit `config/classes.yaml`. Restart the server (or hit `/api/config`) to pick up
changes. Color may be a hex string (`"#22c55e"`) or a `[r, g, b]` float array.

## What's gone vs the older tools

- The standalone `3d-labeler` Vite/React app and the `industrial-point-labeler`
  CLI annotate flow are superseded by Voxa. The 3d-labeler **backend modules**
  are reused (PLY/GLB loading, supervoxels, clustering, RANSAC fitting are
  available for future use); the supervoxel/cluster/RANSAC endpoints aren't
  wired into the new frontend yet but the modules are still present in
  `backend/` if you want to extend the auto-fit pipeline.
- Per-point semantic labels (the old PLY `label`/`instance_id` channels) aren't
  used yet — Voxa is currently cuboid-instance only. The PLY loader still
  reads them so adding a per-point mode later is straightforward.
