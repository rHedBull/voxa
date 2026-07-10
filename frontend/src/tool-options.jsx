import { PresegmentList } from './segment-tools.jsx';
import DrawMode from './draw-mode.jsx';

function AutoConfirmToggle({ tool, autoConfirm, setAutoConfirm }) {
  return (
    <label className="tool-opt-check" title="When on, applying labels a group and marks it confirmed immediately">
      <input type="checkbox" checked={!!autoConfirm[tool]}
        onChange={(e) => setAutoConfirm((m) => ({ ...m, [tool]: e.target.checked }))} />
      Auto-confirm on apply
    </label>
  );
}

function PresegOptions({
  presegRapid, setPresegRapid, setFastPos, autoConfirm, setAutoConfirm,
  segState, setSegState, classes, viewerRef, cloud, promotedSegIds,
}) {
  return (
    <div className="tool-options tool-options-presegment">
      <div className="tool-opt-toggle">
        <button
          className={!presegRapid ? 'active' : ''}
          onClick={() => setPresegRapid(false)}
        >manual</button>
        <button
          className={presegRapid ? 'active' : ''}
          onClick={() => { setFastPos(0); setPresegRapid(true); }}
        >rapid</button>
      </div>
      <AutoConfirmToggle tool="presegment" autoConfirm={autoConfirm} setAutoConfirm={setAutoConfirm} />
      {segState && (
        <PresegmentList
          segState={segState}
          setSegState={setSegState}
          classes={classes}
          viewerRef={viewerRef}
          cloud={cloud}
          excludeSegIds={promotedSegIds}
        />
      )}
    </div>
  );
}

function DrawOptions({
  viewerRef, classes, setSegState, onExit, pointSize, setPointSize,
  activeClass, setActiveClass, onDrawApplied, autoConfirm, setAutoConfirm,
}) {
  return (
    <div className="tool-options tool-options-draw">
      <DrawMode
        viewerRef={viewerRef}
        classes={classes}
        setSegState={setSegState}
        onExit={onExit}
        pointSize={pointSize}
        setPointSize={setPointSize}
        defaultClassId={classes.find((c) => c.id === activeClass)?.class_id ?? classes[0]?.class_id ?? 0}
        onClassChange={(cid) => {
          const cls = classes.find((c) => c.class_id === cid);
          if (cls) setActiveClass(cls.id);
        }}
        onApplied={onDrawApplied}
      />
      <AutoConfirmToggle tool="draw" autoConfirm={autoConfirm} setAutoConfirm={setAutoConfirm} />
    </div>
  );
}

function BoxOptions({ autoConfirm, setAutoConfirm, hasBox, onDrawBox, onApply }) {
  return (
    <div className="tool-options tool-options-box">
      {hasBox ? (
        <div className="tool-opt-toggle">
          <button onClick={onDrawBox}>Clear box</button>
          <button className="active" onClick={onApply}>Apply (Ctrl+Enter)</button>
        </div>
      ) : (
        <>
          <div className="tool-opt-toggle">
            <button className="active" onClick={onDrawBox}>Draw a box</button>
          </div>
          <p className="tool-opt-hint">Draw a box, then transform it to enclose points and Apply.</p>
        </>
      )}
      <AutoConfirmToggle tool="box" autoConfirm={autoConfirm} setAutoConfirm={setAutoConfirm} />
    </div>
  );
}

export default function ToolOptions(props) {
  const { activeTool } = props;
  if (activeTool === 'presegment') return <PresegOptions {...props} />;
  if (activeTool === 'draw') return <DrawOptions {...props} />;
  if (activeTool === 'box') return <BoxOptions {...props} />;
  return null;
}
