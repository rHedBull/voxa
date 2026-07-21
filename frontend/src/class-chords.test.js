import { describe, it, expect } from 'vitest';
import { CLASS_GROUPS, assignable, groupMembers, chordStep } from './class-chords.js';

const CLASSES = [
  { id: 'elbow', label: 'Elbow', hotkey: '2', group: 'pipe-network', frozen: false },
  { id: 'pipe-straight', label: 'Pipe straight', hotkey: '1', group: 'pipe-network', frozen: false },
  { id: 'wall', label: 'Wall', hotkey: '1', group: 'stuff', frozen: false },
  { id: 'pipe', label: 'Pipe (legacy)', hotkey: '', group: 'legacy', frozen: true },
];

describe('class-chords', () => {
  it('has 8 chorded groups + un-chorded legacy', () => {
    expect(CLASS_GROUPS.filter((g) => g.key).map((g) => g.key))
      .toEqual(['1', '2', '3', '4', '5', '6', '7', '8']);
    expect(CLASS_GROUPS.find((g) => g.id === 'legacy').key).toBeNull();
  });

  it('assignable() excludes frozen', () => {
    expect(assignable(CLASSES).map((c) => c.id)).not.toContain('pipe');
  });

  it('first stroke: group key selects the group', () => {
    expect(chordStep(null, '1', CLASSES)).toEqual(
      { type: 'group', group: CLASS_GROUPS[0] });
  });

  it('first stroke: non-group key passes through', () => {
    expect(chordStep(null, 'f', CLASSES)).toEqual({ type: 'pass' });
  });

  it('second stroke: member key classifies within the pending group only', () => {
    const g = CLASS_GROUPS.find((x) => x.id === 'pipe-network');
    const r = chordStep(g, '2', CLASSES);
    expect(r.type).toBe('class');
    expect(r.cls.id).toBe('elbow');
    // '1' in stuff vs pipe-network must not collide:
    expect(chordStep(g, '1', CLASSES).cls.id).toBe('pipe-straight');
  });

  it('second stroke: Escape or invalid key cancels', () => {
    const g = CLASS_GROUPS.find((x) => x.id === 'pipe-network');
    expect(chordStep(g, 'Escape', CLASSES)).toEqual({ type: 'cancel' });
    expect(chordStep(g, 'z', CLASSES)).toEqual({ type: 'cancel' });
  });

  it('groupMembers returns assignable members of one group', () => {
    expect(groupMembers('pipe-network', CLASSES).length).toBe(2);
    expect(groupMembers('legacy', CLASSES)).toEqual([]);
  });
});
