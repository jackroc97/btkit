/**
 * Composite score builder panel.
 *
 * Renders a toggle button + collapsible config panel. The config (label +
 * metric/weight rows) is stored in /api/preferences/{prefKey} and
 * propagated upward via onChange so callers can inject a computed column.
 *
 * Score = Σ(weight_i × metric_i)  (raw weighted sum, no normalisation)
 */

import { useState, useCallback, useEffect, useRef } from 'react'

export interface ScoreWeight {
  metric: string
  weight: number
}

export interface ScoreConfig {
  label: string
  weights: ScoreWeight[]
}

const METRICS: { value: string; label: string }[] = [
  { value: 'sharpe',           label: 'Sharpe' },
  { value: 'sortino',          label: 'Sortino' },
  { value: 'calmar',           label: 'Calmar' },
  { value: 'recovery_factor',  label: 'Recovery Factor' },
  { value: 'cagr',             label: 'CAGR %' },
  { value: 'total_return_pct', label: 'Total Return %' },
  { value: 'win_rate',         label: 'Win Rate' },
  { value: 'profit_factor',    label: 'Profit Factor' },
  { value: 'avg_pnl',          label: 'Avg P&L' },
  { value: 'max_drawdown_pct', label: 'Max DD %' },
  { value: 'n_trades',         label: 'N Trades' },
]

const DEFAULT_CONFIG: ScoreConfig = { label: 'Score', weights: [{ metric: 'sharpe', weight: 1 }] }

interface Props {
  prefKey:  string
  onChange: (config: ScoreConfig | null) => void
}

export default function CompositeScore({ prefKey, onChange }: Props) {
  const [open,   setOpen]   = useState(false)
  const [active, setActive] = useState(false)
  const [config, setConfig] = useState<ScoreConfig>(DEFAULT_CONFIG)
  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const onChangeRef = useRef(onChange)
  onChangeRef.current = onChange

  // Load on mount
  useEffect(() => {
    fetch(`/api/preferences/${prefKey}`)
      .then(r => r.json())
      .then(({ value }: { value: ScoreConfig | null }) => {
        if (value) {
          setConfig(value)
          // Don't auto-activate; user must toggle
        }
      })
      .catch(() => {})
  }, [prefKey])

  const save = useCallback((cfg: ScoreConfig) => {
    if (saveTimer.current) clearTimeout(saveTimer.current)
    saveTimer.current = setTimeout(() => {
      fetch(`/api/preferences/${prefKey}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ value: cfg }),
      }).catch(() => {})
    }, 500)
  }, [prefKey])

  const updateConfig = useCallback((cfg: ScoreConfig) => {
    setConfig(cfg)
    save(cfg)
    if (active) onChangeRef.current(cfg)
  }, [active, save])

  const toggleActive = useCallback(() => {
    setActive(prev => {
      const next = !prev
      onChangeRef.current(next ? config : null)
      return next
    })
  }, [config])

  const setLabel = (label: string) => updateConfig({ ...config, label })

  const setWeight = (i: number, weight: number) => {
    const weights = config.weights.map((w, idx) => idx === i ? { ...w, weight } : w)
    updateConfig({ ...config, weights })
  }

  const setMetric = (i: number, metric: string) => {
    const weights = config.weights.map((w, idx) => idx === i ? { ...w, metric } : w)
    updateConfig({ ...config, weights })
  }

  const addRow = () => updateConfig({
    ...config,
    weights: [...config.weights, { metric: 'sharpe', weight: 1 }],
  })

  const removeRow = (i: number) => updateConfig({
    ...config,
    weights: config.weights.filter((_, idx) => idx !== i),
  })

  const clearAll = () => {
    const cleared: ScoreConfig = { label: 'Score', weights: [] }
    setConfig(cleared)
    save(cleared)
    if (active) {
      onChangeRef.current(null)
      setActive(false)
    }
  }

  const btnBase: React.CSSProperties = {
    borderRadius: 6, cursor: 'pointer', fontSize: '0.72rem',
    fontWeight: 600, padding: '3px 10px', lineHeight: 1.6,
    transition: 'all 0.15s',
  }

  return (
    <div>
      {/* Toggle */}
      <button
        onClick={() => { setOpen(o => !o); if (!open && !active) {} }}
        style={{
          ...btnBase,
          background: active ? 'rgba(99,102,241,0.15)' : 'transparent',
          border: `1px solid ${active ? 'rgba(99,102,241,0.5)' : '#2a3245'}`,
          color: active ? '#818cf8' : '#64748b',
        }}
      >
        <i className="bi bi-calculator me-1" />Σ Score
      </button>

      {/* Panel */}
      {open && (
        <div style={{
          marginTop: 8, padding: '12px 14px',
          background: '#1a2030', border: '1px solid #2a3245',
          borderRadius: 8, minWidth: 340,
        }}>
          {/* Label row */}
          <div className="d-flex align-items-center mb-2" style={{ gap: 8 }}>
            <span style={{ color: '#64748b', fontSize: '0.72rem', minWidth: 44 }}>Label</span>
            <input
              value={config.label}
              onChange={e => setLabel(e.target.value)}
              style={{
                background: '#161b27', border: '1px solid #2a3245',
                borderRadius: 5, color: '#cbd5e1',
                fontSize: '0.78rem', padding: '2px 8px', flex: 1,
              }}
            />
          </div>

          {/* Weight rows */}
          {config.weights.map((w, i) => (
            <div key={i} className="d-flex align-items-center mb-1" style={{ gap: 6 }}>
              <select
                value={w.metric}
                onChange={e => setMetric(i, e.target.value)}
                style={{
                  background: '#161b27', border: '1px solid #2a3245',
                  borderRadius: 5, color: '#cbd5e1',
                  fontSize: '0.75rem', padding: '2px 6px', flex: 1,
                }}
              >
                {METRICS.map(m => (
                  <option key={m.value} value={m.value}>{m.label}</option>
                ))}
              </select>
              <input
                type="number"
                step="0.1"
                value={w.weight}
                onChange={e => setWeight(i, parseFloat(e.target.value) || 0)}
                style={{
                  background: '#161b27', border: '1px solid #2a3245',
                  borderRadius: 5, color: '#cbd5e1',
                  fontSize: '0.75rem', padding: '2px 6px', width: 64, textAlign: 'right',
                }}
              />
              <button
                onClick={() => removeRow(i)}
                style={{
                  background: 'none', border: 'none',
                  color: '#475569', cursor: 'pointer', padding: '2px 4px', lineHeight: 1,
                }}
              >
                <i className="bi bi-x" />
              </button>
            </div>
          ))}

          {/* Actions */}
          <div className="d-flex align-items-center mt-2" style={{ gap: 6 }}>
            <button
              onClick={addRow}
              style={{
                ...btnBase,
                background: 'transparent',
                border: '1px solid #2a3245',
                color: '#64748b',
              }}
            >
              <i className="bi bi-plus me-1" />Add
            </button>
            <button
              onClick={clearAll}
              style={{
                ...btnBase,
                background: 'transparent',
                border: '1px solid rgba(248,113,113,0.3)',
                color: '#f87171',
              }}
            >
              Clear
            </button>
            <div className="ms-auto">
              <button
                onClick={toggleActive}
                style={{
                  ...btnBase,
                  background: active ? 'rgba(99,102,241,0.15)' : 'rgba(99,102,241,0.08)',
                  border: `1px solid ${active ? 'rgba(99,102,241,0.5)' : 'rgba(99,102,241,0.25)'}`,
                  color: active ? '#818cf8' : '#6366f1',
                }}
              >
                {active ? 'Remove column' : 'Apply column'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
