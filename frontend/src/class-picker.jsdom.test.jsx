// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, cleanup, fireEvent, screen } from '@testing-library/react';
import { ClassPickerModal } from './class-picker.jsx';

const CLASSES = [
  { id: 'pipe-straight', label: 'Pipe straight', color: '#22c55e', hotkey: '1', group: 'pipe-network', frozen: false },
  { id: 'elbow', label: 'Elbow', color: '#16a34a', hotkey: '2', group: 'pipe-network', frozen: false },
  { id: 'wall', label: 'Wall', color: '#64748b', hotkey: '1', group: 'stuff', frozen: false },
  { id: 'pipe', label: 'Pipe (legacy)', color: '#4d7c5f', hotkey: '', group: 'legacy', frozen: true },
];

afterEach(cleanup);

describe('ClassPickerModal (two-column master–detail + chords)', () => {
  it('shows groups left and the first group\'s members right; no frozen/legacy', () => {
    render(<ClassPickerModal classes={CLASSES} onPick={() => {}} onClose={() => {}} />);
    expect(screen.getByText('Pipe network')).toBeTruthy();
    expect(screen.getByText('Stuff')).toBeTruthy();
    expect(screen.getByText('Elbow')).toBeTruthy();          // first group's members visible
    expect(screen.queryByText('Wall')).toBeNull();           // other group's members not shown
    expect(screen.queryByText('Pipe (legacy)')).toBeNull();
    expect(screen.queryByText('Legacy (frozen)')).toBeNull();
  });

  it('clicking a group swaps the member column; clicking a member picks it', () => {
    const onPick = vi.fn();
    render(<ClassPickerModal classes={CLASSES} onPick={onPick} onClose={() => {}} />);
    fireEvent.click(screen.getByText('Stuff'));
    expect(screen.getByText('Wall')).toBeTruthy();
    expect(screen.queryByText('Elbow')).toBeNull();
    fireEvent.click(screen.getByText('Wall'));
    expect(onPick.mock.calls[0][0].id).toBe('wall');
  });

  it('two-stroke chord picks the right class within the armed group', () => {
    const onPick = vi.fn();
    render(<ClassPickerModal classes={CLASSES} onPick={onPick} onClose={() => {}} />);
    fireEvent.keyDown(window, { key: '1' });   // arm pipe-network
    fireEvent.keyDown(window, { key: '2' });   // elbow
    expect(onPick).toHaveBeenCalledTimes(1);
    expect(onPick.mock.calls[0][0].id).toBe('elbow');
  });

  it('unarmed digit is a group key even if it matches a visible member key', () => {
    const onPick = vi.fn();
    render(<ClassPickerModal classes={CLASSES} onPick={onPick} onClose={() => {}} />);
    // '8' = stuff group; then '1' must pick Wall (stuff), not Pipe straight.
    fireEvent.keyDown(window, { key: '8' });
    fireEvent.keyDown(window, { key: '1' });
    expect(onPick.mock.calls[0][0].id).toBe('wall');
  });

  it('Esc disarms first, then closes', () => {
    const onClose = vi.fn();
    render(<ClassPickerModal classes={CLASSES} onPick={() => {}} onClose={onClose} />);
    fireEvent.keyDown(window, { key: '1' });
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).not.toHaveBeenCalled();
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});

describe('ClassPickerModal — non-object categories (phase 2)', () => {
  it('renders the four category buttons and picks one by click', () => {
    const onPick = vi.fn();
    render(<ClassPickerModal classes={CLASSES} onPick={onPick} onClose={() => {}} />);
    expect(screen.getByText('Non-object')).toBeTruthy();
    fireEvent.click(screen.getByText('Artifact'));
    expect(onPick.mock.calls[0][0]).toEqual({ category: 'artifact' });
  });

  it('category hotkeys fire only while no group is armed', () => {
    const onPick = vi.fn();
    render(<ClassPickerModal classes={CLASSES} onPick={onPick} onClose={() => {}} />);
    fireEvent.keyDown(window, { key: 'r' });
    expect(onPick.mock.calls[0][0]).toEqual({ category: 'excluded_review' });

    onPick.mockClear();
    fireEvent.keyDown(window, { key: '1' });    // arm pipe-network
    fireEvent.keyDown(window, { key: 'a' });    // NOT a category pick any more
    expect(onPick).not.toHaveBeenCalled();
  });

  it('allowCategories=false hides them entirely (relabel path)', () => {
    const onPick = vi.fn();
    render(<ClassPickerModal classes={CLASSES} onPick={onPick} onClose={() => {}}
                             allowCategories={false} />);
    expect(screen.queryByText('Non-object')).toBeNull();
    fireEvent.keyDown(window, { key: 'a' });
    expect(onPick).not.toHaveBeenCalled();
  });
});
