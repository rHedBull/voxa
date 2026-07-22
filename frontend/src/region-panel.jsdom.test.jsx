// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import RegionPanel from './region-panel.jsx';

afterEach(cleanup);

const regions = [
  { id: 1, name: 'skid A', status: 'draft', prism: { polygon: [[0, 0], [1, 0], [1, 1]], y0: 0, height: 2 } },
  { id: 2, name: 'row B', status: 'eval_grade', prism: { polygon: [[0, 0], [1, 0], [1, 1]], y0: 0, height: 2 },
    accuracy: { p50: 0.004, p90: 0.008, loa: 'LOA40', measured_at: 'x' } },
];
const stats = {
  1: { id: 1, n_points: 200, n_unlabeled: 50, instances: { 7: { inside: 90, total: 100 } } },
  2: { id: 2, n_points: 100, n_unlabeled: 0, instances: {} },
};
const instances = [{ id: 'a', segId: 7, confirmed: true, label: 'pump', cls: 'tank' }];

function renderPanel(over = {}) {
  const props = {
    regions, stats, instances, classes: [{ id: 'tank', label: 'Tank', color: '#123456' }],
    eyes: new Set(), onToggleEye: vi.fn(), onRename: vi.fn(), onDelete: vi.fn(),
    onFlipStatus: vi.fn().mockResolvedValue(undefined), onSelectInstance: vi.fn(),
    ...over,
  };
  render(<RegionPanel {...props} />);
  return props;
}

describe('RegionPanel', () => {
  it('renders rows with status badge and unlabeled %', () => {
    renderPanel();
    expect(screen.getByText('skid A')).toBeTruthy();
    expect(screen.getByText(/25% unlabeled/)).toBeTruthy();     // 50/200
    // Exact matcher: /eval-grade/ would also match the "Mark eval-grade" button.
    expect(screen.getByText('eval-grade')).toBeTruthy();
  });
  it('delete is offered on draft rows only', () => {
    const p = renderPanel();
    const dels = screen.getAllByTitle('Delete region');
    expect(dels).toHaveLength(1);
    fireEvent.click(dels[0]);
    expect(p.onDelete).toHaveBeenCalledWith(1);
  });
  it('mark eval-grade calls onFlipStatus; eval-grade rows offer back-to-draft', () => {
    const p = renderPanel();
    fireEvent.click(screen.getByText('Mark eval-grade'));
    expect(p.onFlipStatus).toHaveBeenCalledWith(1, 'eval_grade');
    fireEvent.click(screen.getByText('Back to draft'));
    expect(p.onFlipStatus).toHaveBeenCalledWith(2, 'draft');
  });
  it('expanding a row lists majority-inside confirmed instances and selects on click', () => {
    const p = renderPanel();
    fireEvent.click(screen.getByText('skid A'));               // expand
    // getByText(/pump/) returns the label span, whose textContent is just
    // "pump" — assert on the member ROW, which also carries the fraction.
    const row = screen.getByText(/pump/).closest('.region-member');
    expect(row.textContent).toMatch(/90\s*%/);
    fireEvent.click(row);
    expect(p.onSelectInstance).toHaveBeenCalledWith(instances[0]);
  });
  it('eye toggle reports the region id', () => {
    const p = renderPanel();
    fireEvent.click(screen.getAllByTitle(/overlay/i)[0]);
    expect(p.onToggleEye).toHaveBeenCalledWith(1);
  });
});
