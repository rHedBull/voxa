// @vitest-environment jsdom
// Regression test for the scene-switch load race: switching away from an
// annotated scene resets activeSessionId (non-null → null), which re-fires
// the load effect; the re-run is skipped as a self-echo, but its cleanup
// must NOT invalidate the in-flight load for the new scene. When it does,
// the response is dropped: the UI sticks on "Loading…" with the old cloud
// until a full page reload.
import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react';
import { test, expect, vi, beforeEach, afterEach } from 'vitest';

vi.mock('./api.js', () => ({
  VoxaAPI: {
    config: vi.fn(async () => ({ classes: [] })),
    scenes: vi.fn(async () => []),
    listPresegs: vi.fn(async () => []),
    load: vi.fn(),
    getAnnotation: vi.fn(async () => ({ instances: [] })),
    segState: vi.fn(async () => null),
    putAnnotation: vi.fn(async () => ({})),
    segSave: vi.fn(async () => ({})),
  },
  getSegmentState: vi.fn(async () => ({})),
}));
// The real mode components pull in three.js + WebGL; the shell under test
// only threads props into them.
vi.mock('./mode-inspect.jsx', () => ({
  InspectMode: ({ cloud, loading }) => (
    <div data-testid="inspect">{loading ? 'loading' : (cloud?.scene ?? 'none')}</div>
  ),
}));
vi.mock('./mode-label.jsx', () => ({ LabelMode: () => null }));
vi.mock('./mode-compare.jsx', () => ({ CompareMode: () => null }));
vi.mock('./mode-edit.jsx', () => ({ EditMode: () => null }));
vi.mock('./mesh-companion.jsx', () => ({ MeshCompanion: () => null }));

import App from './App.jsx';
import { VoxaAPI } from './api.js';

const SCENES = [
  { id: 'annotated/scene-a', name: 'scene-a', tier: 'annotated' },
  { id: 'annotated/scene-b', name: 'scene-b', tier: 'annotated' },
];

function cloudPayload(scene, sessionId) {
  return {
    scene,
    sessionId,
    sessions: [{ session_id: sessionId, name: sessionId }],
    numPoints: 100,
    numPointsTotal: 100,
    numSubsampled: 10,
    positions: new Float32Array(30),
    colors: new Float32Array(30),
    bbox: { min: [0, 0, 0], max: [1, 1, 1] },
    recenterOffset: [0, 0, 0],
    subsampleIdx: null,
    isFromPrelabel: false,
    meshUrl: null,
    classPalette: null,
  };
}

function deferred() {
  let resolve;
  const promise = new Promise((res) => { resolve = res; });
  return { promise, resolve };
}

beforeEach(() => {
  localStorage.setItem('voxa.activeScene', 'annotated/scene-a');
  localStorage.removeItem('voxa.mode');
  vi.spyOn(window, 'confirm').mockReturnValue(true);
  VoxaAPI.scenes.mockResolvedValue(SCENES);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

test('switching scenes applies the new cloud even though the session reset re-fires the load effect', async () => {
  const loadB = deferred();
  VoxaAPI.load.mockImplementation(async (name) => {
    if (name === 'annotated/scene-a') return cloudPayload(name, 'sess-a');
    return loadB.promise;
  });

  render(<App />);

  // Scene A loads and resumes a session — activeSessionId must be non-null
  // for the switch to exercise the reset-to-null re-render.
  await waitFor(() => {
    expect(screen.getByTestId('inspect').textContent).toBe('annotated/scene-a');
  });

  // Switch to scene B via the picker (confirm dialog mocked to OK).
  fireEvent.click(screen.getByRole('button', { name: /annotated\/scene-a/ }));
  fireEvent.click(screen.getByText('scene-b'));
  await waitFor(() => {
    expect(screen.getByTestId('inspect').textContent).toBe('loading');
  });

  // The load for B resolves after the session-reset re-render has run.
  loadB.resolve(cloudPayload('annotated/scene-b', 'sess-b'));

  await waitFor(() => {
    expect(screen.getByTestId('inspect').textContent).toBe('annotated/scene-b');
  });
});
