import { TOOLS, toolAvailable } from './label-tools.js';
import { ToolButton } from './viewport-atoms.jsx';

// The single home for Label-mode tools: a 3-icon strip over the viewport.
export default function ToolRail({ activeTool, onSelect, ctx }) {
  return (
    <div className="tool-rail">
      {TOOLS.map((t) => {
        const ok = toolAvailable(t.id, ctx);
        return (
          <ToolButton key={t.id} mini icon={t.icon}
            label={ok ? t.label : `${t.label} (unavailable for this scan)`}
            active={activeTool === t.id}
            disabled={!ok}
            onClick={() => onSelect(t.id)} />
        );
      })}
    </div>
  );
}
