# Fast Labeling mode — design

Approved 2026-06-04. A queue-driven sub-mode of Label mode for rapidly classifying
preseg segments one at a time (e.g. the 4,911 `sam3_ransac` segments on
`smart_ais_clean`).

## UX flow

- **Fast** toggle button in Label mode's segment tools panel (annotated tier only),
  hotkey `F`. Toggling on computes the queue, highlights queue item #1 in orange,
  and centers the camera on it. Toggling off (button, `F`, or `Esc` with no popup
  open) restores normal Label interaction.
- HUD strip while active: `#k / N · <pts> pts · preseg: <label>` + key hints.
- `→`/`D` next, `←`/`A` previous (`↑`/`W`, `↓`/`S` aliases). Wraps at both ends.
- Number key (classes.yaml hotkeys) → centered confirm popup
  ("Label 12,400 pts as **Pipe**?"). `Enter` confirms, `Esc` cancels.
- On confirm: the segment is promoted to a **confirmed pointset instance**
  (`kind="pointset"`, `segId`, `cls`, `confirmed: true`, `source: "preseg"`) via the
  existing `segApply('reassign')` path, points recolor, and the queue advances.
  Confirmed segments never reappear (matched by `segId` against session instances),
  including across reloads — instances persist in the session JSON.
- Ctrl+Z undo flows through the existing session delta stack; an undone segment
  re-enters the queue.

## Mechanics

- **Queue derivation (client-side, no backend changes):** all preseg segment ids
  from the already-loaded full instance array; sizes via one `bincount` pass;
  sorted largest-first; minus segIds already promoted to instances. O(N) once per
  toggle (~50 ms at 5M points).
- **Highlight:** existing selection-overlay mechanism with a second color
  (orange `0xffa500`) so it does not clash with the yellow click-selection.
- **Camera:** center orbit target on the segment's bounding sphere, distance fit
  so the segment fills ~1/3 of the viewport. Simple jump (no tween) if no tween
  helper exists.
- **Guards:** Fast mode forces orbit nav (WASD conflicts with walk); keys ignored
  while a text input has focus (same pattern as Label's hotkey handler).

## Testing

- Vitest: queue derivation (sort, promoted-segId exclusion, wrap navigation) as
  exported pure functions.
- Backend untouched → pytest unaffected.
- Browser verification of the full loop on `smart_ais_clean` + `sam3_ransac`.

## Out of scope

- Class suggestions / auto-accept, batch operations, queue filters, tweened camera
  flights, anything touching the backend.
