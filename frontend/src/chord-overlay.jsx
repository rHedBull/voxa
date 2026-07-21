// Transient overlay after the first chord stroke: lists the pending group's
// members with their second keys. Purely presentational.
import { groupMembers } from './class-chords.js';

export function ChordOverlay({ group, classes }) {
  if (!group) return null;
  return (
    <div className="chord-overlay">
      <div className="chord-title">{group.key} — {group.label}</div>
      {groupMembers(group.id, classes).map((c) => (
        <div key={c.id} className="chord-row">
          <span className="chord-key">{c.hotkey}</span>
          <span className="class-swatch" style={{ background: c.color }} />
          <span>{c.label}</span>
        </div>
      ))}
      <div className="chord-hint">Esc to cancel</div>
    </div>
  );
}
