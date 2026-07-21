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

describe('ClassPickerModal (grouped + chords)', () => {
  it('renders group headers and omits frozen classes entirely', () => {
    render(<ClassPickerModal classes={CLASSES} onPick={() => {}} onClose={() => {}} />);
    expect(screen.getByText('Pipe network')).toBeTruthy();
    expect(screen.getByText('Stuff')).toBeTruthy();
    expect(screen.queryByText('Pipe (legacy)')).toBeNull();
    expect(screen.queryByText('Legacy (frozen)')).toBeNull();
  });

  it('two-stroke chord picks the right class within the group', () => {
    const onPick = vi.fn();
    render(<ClassPickerModal classes={CLASSES} onPick={onPick} onClose={() => {}} />);
    fireEvent.keyDown(window, { key: '1' });   // pipe-network group
    fireEvent.keyDown(window, { key: '2' });   // elbow
    expect(onPick).toHaveBeenCalledTimes(1);
    expect(onPick.mock.calls[0][0].id).toBe('elbow');
  });

  it('first Esc backs out of a pending group, second closes', () => {
    const onClose = vi.fn();
    render(<ClassPickerModal classes={CLASSES} onPick={() => {}} onClose={onClose} />);
    fireEvent.keyDown(window, { key: '1' });
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).not.toHaveBeenCalled();
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('clicking a row picks the class', () => {
    const onPick = vi.fn();
    render(<ClassPickerModal classes={CLASSES} onPick={onPick} onClose={() => {}} />);
    fireEvent.click(screen.getByText('Wall'));
    expect(onPick.mock.calls[0][0].id).toBe('wall');
  });
});
