// @vitest-environment jsdom
import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import { test, expect, vi, afterEach } from 'vitest';
import { ContextMenu } from './context-menu.jsx';

// No global test setupFile in this repo (pure-fn suite up to now) — clean up
// the jsdom DOM between tests locally rather than adding one.
afterEach(cleanup);

test('renders items and calls onSelect, closes on outside click', () => {
  const onSelect = vi.fn();
  const onClose = vi.fn();
  render(
    <ContextMenu x={10} y={10} items={[{ label: 'Edit selection…', onSelect, disabled: false }]} onClose={onClose} />
  );
  fireEvent.click(screen.getByText('Edit selection…'));
  expect(onSelect).toHaveBeenCalled();
});

test('disabled item does not call onSelect', () => {
  const onSelect = vi.fn();
  render(
    <ContextMenu x={10} y={10} items={[{ label: 'Edit selection…', onSelect, disabled: true }]} onClose={() => {}} />
  );
  fireEvent.click(screen.getByText('Edit selection…'));
  expect(onSelect).not.toHaveBeenCalled();
});

test('clicking an enabled item also calls onClose', () => {
  const onSelect = vi.fn();
  const onClose = vi.fn();
  render(
    <ContextMenu x={10} y={10} items={[{ label: 'Edit selection…', onSelect, disabled: false }]} onClose={onClose} />
  );
  fireEvent.click(screen.getByText('Edit selection…'));
  expect(onClose).toHaveBeenCalled();
});
