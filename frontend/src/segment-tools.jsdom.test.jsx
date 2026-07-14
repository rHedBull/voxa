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
