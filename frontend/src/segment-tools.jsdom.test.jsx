// @vitest-environment jsdom
// jsdom coverage for PresegmentList's right-click "Edit selection…" wiring
// (Task 10 gap) — same pattern as sam-segment-list.jsdom.test.jsx.
import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import { test, expect, vi, afterEach } from 'vitest';
import { PresegmentList } from './segment-tools.jsx';

afterEach(cleanup);

const classes = [{ class_id: 0, label: 'pipe', color: '#ff0000' }];

function makeSegState(selection) {
  return {
    summary: new Map([
      [1, { classId: 0, nPoints: 100 }],
      [2, { classId: -1, nPoints: 50 }],
    ]),
    selection,
  };
}

test('right-clicking a row opens the context menu with "Edit selection…"', () => {
  const segState = makeSegState(new Set());
  render(<PresegmentList segState={segState} setSegState={() => {}} classes={classes} viewerRef={{ current: null }} cloud={null} />);
  fireEvent.contextMenu(screen.getByText('#1'));
  expect(screen.getByText('Edit selection…')).toBeTruthy();
});

test('empty selection -> menu item is disabled (matches cutEligibility)', () => {
  const segState = makeSegState(new Set());
  render(<PresegmentList segState={segState} setSegState={() => {}} classes={classes} viewerRef={{ current: null }} cloud={null} />);
  fireEvent.contextMenu(screen.getByText('#1'));
  const item = screen.getByText('Edit selection…');
  expect(item.className).toContain('disabled');
});

test('non-empty selection -> menu item is enabled (matches cutEligibility)', () => {
  const segState = makeSegState(new Set([1]));
  render(<PresegmentList segState={segState} setSegState={() => {}} classes={classes} viewerRef={{ current: null }} cloud={null} />);
  fireEvent.contextMenu(screen.getByText('#1'));
  const item = screen.getByText('Edit selection…');
  expect(item.className).not.toContain('disabled');
});

test('clicking the enabled item invokes onEditSelection with {kind:preseg,segId} sources', () => {
  const segState = makeSegState(new Set([1]));
  const onEditSelection = vi.fn();
  render(<PresegmentList segState={segState} setSegState={() => {}} classes={classes}
    viewerRef={{ current: null }} cloud={null} onEditSelection={onEditSelection} />);
  fireEvent.contextMenu(screen.getByText('#1'));
  fireEvent.click(screen.getByText('Edit selection…'));
  expect(onEditSelection).toHaveBeenCalledWith([{ kind: 'preseg', segId: 1 }]);
});

// ── source:'preseg' cut candidates (segState.samSegments) ──────────────────
// These live in the mutable SAM-candidate layer, not segState.instanceFull —
// PresegmentList must surface them as a visually distinct second section,
// backed by samSelection, not segState.selection.

function makeSegStateWithCandidates(samSegments, samSelection = new Set()) {
  return {
    summary: new Map([[1, { classId: 0, nPoints: 100 }]]),
    selection: new Set(),
    samSegments,
    samSelection,
  };
}

test('a source:preseg cut candidate renders as a distinct "Cut #" row, not merged into the Presegments list', () => {
  const segState = makeSegStateWithCandidates(new Map([
    [0, { nPoints: 6723, source: 'preseg' }],
  ]));
  render(<PresegmentList segState={segState} setSegState={() => {}} classes={classes}
    viewerRef={{ current: null }} cloud={null} />);
  expect(screen.getByText('Cut #0')).toBeTruthy();
  // Real presegment #1 keeps its own bare "#1" label — no id collision even
  // though a candidate can share the same numeric id (separate counter).
  expect(screen.getByText('#1')).toBeTruthy();
});

test('a source:sam candidate does not appear in PresegmentList', () => {
  const segState = makeSegStateWithCandidates(new Map([
    [0, { nPoints: 500, source: 'sam' }],
  ]));
  render(<PresegmentList segState={segState} setSegState={() => {}} classes={classes}
    viewerRef={{ current: null }} cloud={null} />);
  expect(screen.queryByText('Cut #0')).toBeNull();
});

test('Ctrl-clicking a cut-candidate row toggles samSelection, not segState.selection', () => {
  const segState = makeSegStateWithCandidates(new Map([
    [0, { nPoints: 6723, source: 'preseg' }],
  ]));
  const setSegState = vi.fn();
  render(<PresegmentList segState={segState} setSegState={setSegState} classes={classes}
    viewerRef={{ current: null }} cloud={null} />);
  fireEvent.click(screen.getByText('Cut #0'), { ctrlKey: true });
  expect(setSegState).toHaveBeenCalledTimes(1);
  const next = setSegState.mock.calls[0][0](segState);
  expect(Array.from(next.samSelection)).toEqual([0]);
  expect(next.selection).toBe(segState.selection); // untouched
});

test('right-clicking a selected cut-candidate row opens an enabled "Edit selection…" wired to kind:sam', () => {
  const segState = makeSegStateWithCandidates(
    new Map([[0, { nPoints: 6723, source: 'preseg' }]]),
    new Set([0]),
  );
  const onEditSelection = vi.fn();
  render(<PresegmentList segState={segState} setSegState={() => {}} classes={classes}
    viewerRef={{ current: null }} cloud={null} onEditSelection={onEditSelection} />);
  fireEvent.contextMenu(screen.getByText('Cut #0'));
  const item = screen.getByText('Edit selection…');
  expect(item.className).not.toContain('disabled');
  fireEvent.click(item);
  // kind:'sam', NOT 'preseg' — the candidate's points live in sam_ids, so a
  // recursive cut must resolve against that array, not preseg_ids.
  expect(onEditSelection).toHaveBeenCalledWith([{ kind: 'sam', segId: 0 }]);
});
