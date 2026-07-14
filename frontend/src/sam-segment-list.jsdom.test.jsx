// @vitest-environment jsdom
// jsdom coverage for the SamSegmentList right-click "Edit selection…" wiring
// (Task 10 gap): the pure toggleSamSelection() helper already has coverage
// in sam-segment-list.test.js — this file exercises the actual row
// contextmenu -> ContextMenu -> cutEligibility wiring end to end.
import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import { test, expect, vi, afterEach } from 'vitest';
import { SamSegmentList } from './sam-segment-list.jsx';

afterEach(cleanup);

function makeSegState(samSelection) {
  return {
    samSegments: new Map([
      [1, { nPoints: 100 }],
      [2, { nPoints: 50 }],
    ]),
    samSelection,
  };
}

test('right-clicking a row opens the context menu with "Edit selection…"', () => {
  const segState = makeSegState(new Set());
  render(<SamSegmentList segState={segState} setSegState={() => {}} />);
  fireEvent.contextMenu(screen.getByText('SAM #1'));
  expect(screen.getByText('Edit selection…')).toBeTruthy();
});

test('empty samSelection -> menu item is disabled (matches cutEligibility)', () => {
  const segState = makeSegState(new Set());
  render(<SamSegmentList segState={segState} setSegState={() => {}} />);
  fireEvent.contextMenu(screen.getByText('SAM #1'));
  const item = screen.getByText('Edit selection…');
  expect(item.className).toContain('disabled');
});

test('non-empty samSelection -> menu item is enabled (matches cutEligibility)', () => {
  const segState = makeSegState(new Set([1]));
  render(<SamSegmentList segState={segState} setSegState={() => {}} />);
  fireEvent.contextMenu(screen.getByText('SAM #1'));
  const item = screen.getByText('Edit selection…');
  expect(item.className).not.toContain('disabled');
});

test('clicking the enabled item invokes onEditSelection with {kind:sam,segId} sources', () => {
  const segState = makeSegState(new Set([1]));
  const onEditSelection = vi.fn();
  render(<SamSegmentList segState={segState} setSegState={() => {}} onEditSelection={onEditSelection} />);
  fireEvent.contextMenu(screen.getByText('SAM #1'));
  fireEvent.click(screen.getByText('Edit selection…'));
  expect(onEditSelection).toHaveBeenCalledWith([{ kind: 'sam', segId: 1 }]);
});
