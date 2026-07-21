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

describe('ClassPickerModal (two-level drill-down + chords)', () => {
  it('level 1 shows group rows only — no class rows, no frozen/legacy', () => {
    render(<ClassPickerModal classes={CLASSES} onPick={() => {}} onClose={() => {}} />);
    expect(screen.getByText('Pipe network')).toBeTruthy();
    expect(screen.getByText('Stuff')).toBeTruthy();
    expect(screen.queryByText('Elbow')).toBeNull();          // members hidden until drill-down
    expect(screen.queryByText('Pipe (legacy)')).toBeNull();
    expect(screen.queryByText('Legacy (frozen)')).toBeNull();
  });

  it('clicking a group opens its member list; clicking a member picks it', () => {
    const onPick = vi.fn();
    render(<ClassPickerModal classes={CLASSES} onPick={onPick} onClose={() => {}} />);
    fireEvent.click(screen.getByText('Pipe network'));
    expect(screen.getByText('Elbow')).toBeTruthy();
    expect(screen.queryByText('Stuff')).toBeNull();          // other groups hidden
    fireEvent.click(screen.getByText('Elbow'));
    expect(onPick.mock.calls[0][0].id).toBe('elbow');
  });

  it('two-stroke chord picks the right class within the group', () => {
    const onPick = vi.fn();
    render(<ClassPickerModal classes={CLASSES} onPick={onPick} onClose={() => {}} />);
    fireEvent.keyDown(window, { key: '1' });   // pipe-network group
    expect(screen.getByText('Pipe straight')).toBeTruthy();  // drilled in
    fireEvent.keyDown(window, { key: '2' });   // elbow
    expect(onPick).toHaveBeenCalledTimes(1);
    expect(onPick.mock.calls[0][0].id).toBe('elbow');
  });

  it('Esc backs out one level, then closes; back row returns to groups', () => {
    const onClose = vi.fn();
    render(<ClassPickerModal classes={CLASSES} onPick={() => {}} onClose={onClose} />);
    fireEvent.keyDown(window, { key: '1' });
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).not.toHaveBeenCalled();
    expect(screen.getByText('Stuff')).toBeTruthy();          // back at group level
    fireEvent.click(screen.getByText('Pipe network'));
    fireEvent.click(screen.getAllByText('Pipe network')[0]); // back row label
    expect(screen.getByText('Stuff')).toBeTruthy();
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
