import { useState, useMemo, useEffect, useCallback } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { type ColDef, type ICellRendererParams, type ValueGetterParams } from 'ag-grid-community'
import BtkAgGrid from '../components/BtkAgGrid'
import CompositeScore, { type ScoreConfig } from '../components/CompositeScore'
import { TagFilterBar } from '../tags/TagFilterBar'
import { TagPicker } from '../tags/TagPicker'
import { TagPillList } from '../tags/TagPill'
import type { Tag } from '../tags/TagsContext'

// ── Types ────────────────────────────────────────────────────────────────────

type Status = 'completed' | 'running' | 'error'
type ViewMode = 'list' | 'grid'

interface ApiStudy {
  id: number
  name: string
  note: string | null
  total_combinations: number
  created_at: string
  finished_at: string | null
  n_backtests: number
  n_completed: number
  n_failed: number
  n_running: number
  strategies: string[]
  strategy_labels: string[]
  sweep_axes: string[]
  data_start: string | null
  data_end: string | null
  total_trades: number
  best_sharpe: number | null
  best_return_pct: number | null
  tags: Tag[]
}

interface ApiBacktest {
  id: number
  study_id: number
  combination_id: number
  strategy_name: string
  strategy_label: string
  initial_equity: number | null
  status: Status
  created_at: string
  n_trades: number
  total_pnl: number
  avg_pnl: number | null
  win_rate: number | null
  start_date: string | null
  end_date: string | null
  sharpe: number | null
  sortino: number | null
  calmar: number | null
  total_return_pct: number | null
  max_drawdown: number | null
  max_drawdown_pct: number | null
  cagr: number | null
  recovery_factor: number | null
  tags: Tag[]
  params: {
    delta?: number | null
    take_profit_pct?: number | null
    stop_loss?: number | null
    min_credit?: number | null
  }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function studyStatus(s: ApiStudy): Status {
  if (s.n_running > 0) return 'running'
  if (s.n_completed > 0) return 'completed'
  if (s.finished_at) return 'error'
  return 'running'
}

function fmtDate(iso: string | null): string {
  if (!iso) return '—'
  const d = new Date(iso)
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}

function fmtDateShort(iso: string | null): string {
  if (!iso) return '—'
  const d = new Date(iso)
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: '2-digit' })
}

function backtestNote(b: ApiBacktest): string | null {
  const parts: string[] = []
  const p = b.params ?? {}
  if (p.delta != null) parts.push(`δ ${p.delta.toFixed(2)}`)
  if (p.take_profit_pct != null) parts.push(`TP ${(p.take_profit_pct * 100).toFixed(0)}%`)
  if (p.stop_loss != null) parts.push(`SL ${p.stop_loss}`)
  if (p.min_credit != null) parts.push(`min $${p.min_credit}`)
  return parts.length ? parts.join(' · ') : null
}

function sharpeColor(v: number): string {
  if (v >= 1.5) return 'positive'
  if (v >= 0.8) return 'amber'
  return 'negative'
}

function fmt(v: number | null, prefix = '', suffix = '', decimals = 2): string {
  if (v === null) return '—'
  const s = Math.abs(v).toFixed(decimals)
  return `${prefix}${v < 0 ? '-' : ''}${s}${suffix}`
}

function StatusBadge({ status }: { status: Status }) {
  const label = status === 'running' ? (
    <><i className="bi bi-arrow-repeat me-1" style={{ animation: 'spin 1.2s linear infinite', display: 'inline-block' }} />running</>
  ) : status
  return <span className={`btk-badge ${status}`}>{label}</span>
}

function StatusCell(p: { value: Status }) {
  const s = p.value
  if (s === 'running') return (
    <span className="btk-badge running">
      <i className="bi bi-arrow-repeat me-1" style={{ animation: 'spin 1.2s linear infinite', display: 'inline-block' }} />
      running
    </span>
  )
  return <span className={`btk-badge ${s}`}>{s}</span>
}

// ── Grid column definitions ───────────────────────────────────────────────────

function makeStudyCols(): ColDef<ApiStudy>[] {
  return [
    {
      field: 'name', headerName: 'Name', minWidth: 160, flex: 2,
      cellStyle: { color: '#e2e8f0', fontWeight: 600 },
    },
    {
      headerName: 'Strategy', minWidth: 110, flex: 1,
      valueGetter: p => (p.data?.strategy_labels?.length ? p.data.strategy_labels : p.data?.strategies ?? []).join(', '),
      cellStyle: { color: '#60a5fa', fontFamily: 'monospace', fontSize: '0.78rem' },
    },
    {
      headerName: 'Tags', width: 160,
      cellRenderer: (p: ICellRendererParams<ApiStudy>) => {
        if (!p.data) return null
        const tags = p.data.tags ?? []
        return (
          <div style={{ display: 'flex', alignItems: 'center', gap: 4, height: '100%' }}>
            <TagPillList tags={tags} maxVisible={2} />
          </div>
        )
      },
    },
    {
      headerName: 'Combs', width: 80,
      valueGetter: p => p.data ? `${p.data.n_completed}/${p.data.total_combinations}` : '',
      cellStyle: { color: 'var(--btk-muted-dk)' },
    },
    {
      headerName: 'Status', width: 104,
      valueGetter: p => p.data ? studyStatus(p.data) : '',
      cellRenderer: StatusCell,
    },
    {
      field: 'best_sharpe', headerName: 'Best Sharpe', width: 108,
      valueFormatter: p => p.value != null ? (p.value as number).toFixed(2) : '—',
      cellStyle: p => {
        const v = p.value as number | null
        if (v == null) return null
        return { color: v >= 1.5 ? '#4ade80' : v >= 0.8 ? '#f59e0b' : '#f87171', fontWeight: 600 }
      },
    },
    {
      field: 'best_return_pct', headerName: 'Best Return', width: 110,
      valueFormatter: p => { const v = p.value as number | null; return v == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(1)}%` },
      cellStyle: p => ({ color: ((p.value as number | null) ?? 0) >= 0 ? '#4ade80' : '#f87171' }),
    },
    {
      field: 'total_trades', headerName: 'Trades', width: 80,
      valueFormatter: p => p.value != null ? (p.value as number).toLocaleString() : '—',
    },
    {
      headerName: 'Period', minWidth: 150,
      valueGetter: p => p.data?.data_start && p.data?.data_end
        ? `${fmtDateShort(p.data.data_start)} – ${fmtDateShort(p.data.data_end)}`
        : '—',
      cellStyle: { color: 'var(--btk-muted-dk)' },
    },
    {
      field: 'created_at', headerName: 'Created', width: 110,
      valueFormatter: p => fmtDate(p.value as string | null),
      cellStyle: { color: 'var(--btk-muted-dk)' },
    },
  ]
}

function makeBacktestCols(
  onTagsChanged: (backtestId: number, tags: Tag[]) => void,
): ColDef<ApiBacktest>[] {
  return [
    {
      field: 'strategy_label', headerName: 'Strategy', minWidth: 140, flex: 1,
      cellStyle: { color: '#60a5fa', fontFamily: 'monospace', fontSize: '0.78rem' },
    },
    {
      headerName: 'Params', minWidth: 120, flex: 1,
      valueGetter: p => p.data ? (backtestNote(p.data) ?? '') : '',
      cellStyle: { color: '#94a3b8', fontFamily: 'ui-monospace, monospace', fontSize: '0.78rem' },
    },
    {
      headerName: 'Tags', width: 180,
      cellRenderer: (p: ICellRendererParams<ApiBacktest>) => {
        if (!p.data) return null
        return (
          <div style={{ display: 'flex', alignItems: 'center', gap: 4, height: '100%' }} onClick={e => e.stopPropagation()}>
            <TagPillList tags={p.data.tags ?? []} maxVisible={2} />
            <TagPicker
              backtestId={p.data.id}
              currentTags={p.data.tags ?? []}
              onChanged={next => onTagsChanged(p.data!.id, next)}
            />
          </div>
        )
      },
    },
    {
      field: 'status', headerName: 'Status', width: 104,
      cellRenderer: StatusCell,
    },
    {
      field: 'sharpe', headerName: 'Sharpe', width: 90,
      valueFormatter: p => p.value != null ? (p.value as number).toFixed(2) : '—',
      cellStyle: p => {
        const v = p.value as number | null
        if (v == null) return null
        return { color: v >= 1.5 ? '#4ade80' : v >= 0.8 ? '#f59e0b' : '#f87171', fontWeight: 600 }
      },
    },
    {
      field: 'sortino', headerName: 'Sortino', width: 90,
      valueFormatter: p => p.value != null ? (p.value as number).toFixed(2) : '—',
      cellStyle: p => {
        const v = p.value as number | null
        if (v == null) return null
        return { color: v >= 1.5 ? '#4ade80' : v >= 0.8 ? '#f59e0b' : '#f87171', fontWeight: 600 }
      },
    },
    {
      field: 'calmar', headerName: 'Calmar', width: 90,
      valueFormatter: p => p.value != null ? (p.value as number).toFixed(2) : '—',
      cellStyle: p => {
        const v = p.value as number | null
        if (v == null) return null
        return { color: v >= 1 ? '#4ade80' : v >= 0.5 ? '#f59e0b' : '#f87171' }
      },
    },
    {
      field: 'total_return_pct', headerName: 'Return', width: 90,
      valueFormatter: p => { const v = p.value as number | null; return v == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(1)}%` },
      cellStyle: p => ({ color: ((p.value as number | null) ?? 0) >= 0 ? '#4ade80' : '#f87171' }),
    },
    {
      field: 'recovery_factor', headerName: 'Recovery', width: 100,
      valueFormatter: p => p.value != null ? (p.value as number).toFixed(2) : '—',
      cellStyle: p => {
        const v = p.value as number | null
        if (v == null) return null
        return { color: v >= 1 ? '#4ade80' : '#f87171' }
      },
    },
    {
      field: 'win_rate', headerName: 'Win Rate', width: 90,
      valueFormatter: p => p.value != null ? `${((p.value as number) * 100).toFixed(1)}%` : '—',
    },
    {
      field: 'n_trades', headerName: 'Trades', width: 80,
    },
    {
      headerName: 'Period', minWidth: 150,
      valueGetter: p => p.data?.start_date && p.data?.end_date
        ? `${fmtDateShort(p.data.start_date)} – ${fmtDateShort(p.data.end_date)}`
        : '—',
      cellStyle: { color: 'var(--btk-muted-dk)' },
    },
    {
      field: 'created_at', headerName: 'Date Ran', width: 110,
      valueFormatter: p => fmtDate(p.value as string | null),
      cellStyle: { color: 'var(--btk-muted-dk)' },
    },
  ]
}

// ── List-view row components ─────────────────────────────────────────────────

function StudyRow({ s }: { s: ApiStudy }) {
  const status = studyStatus(s)
  const labels = s.strategy_labels.join(', ') || s.strategies.join(', ') || '—'
  return (
    <Link to={`/study/${s.id}`} className="btk-list-item">
      <div className="d-flex justify-content-between align-items-start gap-3">
        <div className="flex-grow-1 min-width-0">
          <div className="btk-item-name d-flex align-items-center gap-2 flex-wrap">
            {s.name}
            {s.tags?.length > 0 && <TagPillList tags={s.tags} maxVisible={3} />}
          </div>
          <div className="btk-item-sub">
            <span className="me-2" style={{ color: '#60a5fa', fontFamily: 'monospace', fontSize: '0.75rem' }}>
              {labels}
            </span>
            <span className="me-1 text-secondary">·</span>
            <span>{s.total_combinations} combinations</span>
            <span className="mx-2 text-secondary">·</span>
            <span>{fmtDate(s.created_at)}</span>
          </div>
        </div>
        <div className="d-flex align-items-center gap-2 flex-shrink-0">
          <StatusBadge status={status} />
          <i className="bi bi-chevron-right btk-chevron" />
        </div>
      </div>
      {status === 'completed' && s.best_sharpe !== null && (
        <div className="btk-metrics">
          <div className="btk-metric">
            <div className="btk-metric-label">Best Sharpe</div>
            <div className={`btk-metric-value ${sharpeColor(s.best_sharpe)}`}>{s.best_sharpe.toFixed(2)}</div>
          </div>
          <div className="btk-metric">
            <div className="btk-metric-label">Best Return</div>
            <div className={`btk-metric-value ${s.best_return_pct! >= 0 ? 'positive' : 'negative'}`}>
              {fmt(s.best_return_pct, '+', '%')}
            </div>
          </div>
          <div className="btk-metric">
            <div className="btk-metric-label">Combinations</div>
            <div className="btk-metric-value neutral">{s.n_completed}/{s.total_combinations}</div>
          </div>
          <div className="btk-metric">
            <div className="btk-metric-label">Trades</div>
            <div className="btk-metric-value neutral">{s.total_trades.toLocaleString()}</div>
          </div>
          {s.data_start && s.data_end && (
            <div className="btk-metric">
              <div className="btk-metric-label">Period</div>
              <div className="btk-metric-value neutral" style={{ fontSize: '0.7rem' }}>
                {fmtDateShort(s.data_start)} – {fmtDateShort(s.data_end)}
              </div>
            </div>
          )}
        </div>
      )}
    </Link>
  )
}

function BacktestRow({
  b,
  onTagsChanged,
}: {
  b: ApiBacktest
  onTagsChanged: (id: number, tags: Tag[]) => void
}) {
  const note = backtestNote(b)
  const winPct = b.win_rate != null ? b.win_rate * 100 : null
  return (
    <Link to={`/backtest/${b.id}`} className="btk-list-item">
      <div className="d-flex justify-content-between align-items-start gap-3">
        <div className="flex-grow-1 min-width-0">
          <div className="btk-item-name d-flex align-items-center gap-2 flex-wrap">
            {b.strategy_label}
            {b.tags?.length > 0 && <TagPillList tags={b.tags} maxVisible={3} />}
          </div>
          {note && <div className="btk-item-note">{note}</div>}
          {!note && <div className="btk-item-sub">{fmtDate(b.created_at)}</div>}
        </div>
        <div className="d-flex align-items-center gap-2 flex-shrink-0" onClick={e => e.preventDefault()}>
          <TagPicker
            backtestId={b.id}
            currentTags={b.tags ?? []}
            onChanged={next => onTagsChanged(b.id, next)}
          />
          {note && <span className="btk-item-sub flex-shrink-0">{fmtDate(b.created_at)}</span>}
          <StatusBadge status={b.status} />
          <i className="bi bi-chevron-right btk-chevron" />
        </div>
      </div>
      {b.status === 'completed' && b.sharpe !== null && (
        <div className="btk-metrics">
          <div className="btk-metric">
            <div className="btk-metric-label">Sharpe</div>
            <div className={`btk-metric-value ${sharpeColor(b.sharpe)}`}>{b.sharpe.toFixed(2)}</div>
          </div>
          <div className="btk-metric">
            <div className="btk-metric-label">Return</div>
            <div className={`btk-metric-value ${b.total_return_pct! >= 0 ? 'positive' : 'negative'}`}>
              {fmt(b.total_return_pct, b.total_return_pct! >= 0 ? '+' : '', '%')}
            </div>
          </div>
          <div className="btk-metric">
            <div className="btk-metric-label">Win Rate</div>
            <div className="btk-metric-value neutral">{fmt(winPct, '', '%', 1)}</div>
          </div>
          <div className="btk-metric">
            <div className="btk-metric-label">Trades</div>
            <div className="btk-metric-value neutral">{b.n_trades}</div>
          </div>
        </div>
      )}
    </Link>
  )
}

// ── View toggle ───────────────────────────────────────────────────────────────

function ViewToggle({ mode, onChange }: { mode: ViewMode; onChange: (m: ViewMode) => void }) {
  return (
    <div className="d-flex" style={{ background: '#1e2535', borderRadius: 6, padding: 2, gap: 2 }}>
      {([
        { m: 'list' as ViewMode, icon: 'bi-list-ul',   title: 'List view'  },
        { m: 'grid' as ViewMode, icon: 'bi-grid-3x3',  title: 'Grid view'  },
      ]).map(({ m, icon, title }) => (
        <button key={m} title={title} onClick={() => onChange(m)} style={{
          background: mode === m ? '#2563eb' : 'transparent',
          border: 'none', borderRadius: 4,
          color: mode === m ? '#fff' : '#64748b',
          cursor: 'pointer', padding: '5px 9px', lineHeight: 1,
          transition: 'background 0.15s, color 0.15s',
        }}>
          <i className={`bi ${icon}`} style={{ fontSize: '0.9rem' }} />
        </button>
      ))}
    </div>
  )
}

// ── Page ─────────────────────────────────────────────────────────────────────

type Tab = 'studies' | 'backtests'

export default function Index() {
  const navigate = useNavigate()
  const [tab, setTab]           = useState<Tab>('studies')
  const [viewMode, setViewMode] = useState<ViewMode>('list')
  const [query, setQuery]       = useState('')
  const [studies, setStudies]   = useState<ApiStudy[]>([])
  const [backtests, setBacktests] = useState<ApiBacktest[]>([])
  const [loading, setLoading]   = useState(true)
  const [error, setError]       = useState<string | null>(null)
  const [activeTagFilter, setActiveTagFilter] = useState<Set<number>>(new Set())

  useEffect(() => {
    Promise.all([
      fetch('/api/studies').then(r => r.json()),
      fetch('/api/backtests').then(r => r.json()),
    ])
      .then(([s, b]) => { setStudies(s); setBacktests(b) })
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false))
  }, [])

  const updateBacktestTags = useCallback((id: number, tags: Tag[]) => {
    setBacktests(prev => prev.map(b => b.id === id ? { ...b, tags } : b))
  }, [])


  // Collect all tags visible in the current tab for the filter bar
  const tagsInView = useMemo((): Tag[] => {
    const seen = new Map<number, Tag>()
    const source = tab === 'studies' ? studies : backtests
    for (const item of source) {
      for (const tag of (item.tags ?? [])) {
        if (!seen.has(tag.id)) seen.set(tag.id, tag)
      }
    }
    return [...seen.values()].sort((a, b) => a.name.localeCompare(b.name))
  }, [tab, studies, backtests])

  const filteredStudies = useMemo(() => {
    const q = query.toLowerCase()
    return studies.filter(s => {
      const matchesQuery = (
        (s.name ?? '').toLowerCase().includes(q) ||
        (s.strategies ?? []).some(st => (st ?? '').toLowerCase().includes(q)) ||
        (s.strategy_labels ?? []).some(l => (l ?? '').toLowerCase().includes(q))
      )
      const matchesTags = activeTagFilter.size === 0 ||
        (s.tags ?? []).some(t => activeTagFilter.has(t.id))
      return matchesQuery && matchesTags
    })
  }, [query, studies, activeTagFilter])

  const filteredBacktests = useMemo(() => {
    const q = query.toLowerCase()
    return backtests.filter(b => {
      const matchesQuery = (
        (b.strategy_label ?? '').toLowerCase().includes(q) ||
        (b.strategy_name ?? '').toLowerCase().includes(q)
      )
      const matchesTags = activeTagFilter.size === 0 ||
        (b.tags ?? []).some(t => activeTagFilter.has(t.id))
      return matchesQuery && matchesTags
    })
  }, [query, backtests, activeTagFilter])

  const [scoreConfig, setScoreConfig] = useState<ScoreConfig | null>(null)

  // Memoize col defs so they don't recreate on every render
  const studyCols  = useMemo(() => makeStudyCols(), [])
  const backtestCols = useMemo(() => makeBacktestCols(updateBacktestTags), [updateBacktestTags])

  const finalBacktestCols = useMemo((): ColDef<ApiBacktest>[] => {
    if (!scoreConfig || scoreConfig.weights.length === 0) return backtestCols
    return [...backtestCols, {
      headerName: scoreConfig.label || 'Score',
      colId: '__score__',
      valueGetter: (p: ValueGetterParams<ApiBacktest>) => {
        if (!p.data) return null
        let sum = 0
        for (const { metric, weight } of scoreConfig.weights) {
          const v = (p.data as unknown as Record<string, number | null>)[metric]
          if (v != null) sum += weight * v
        }
        return sum
      },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      valueFormatter: (p: any) => p.value != null ? (p.value as number).toFixed(3) : '—',
      cellStyle: { color: '#818cf8', fontWeight: 600 },
      sortable: true,
    } as ColDef<ApiBacktest>]
  }, [backtestCols, scoreConfig])

  const items = tab === 'studies' ? filteredStudies : filteredBacktests
  const placeholder = tab === 'studies' ? 'Search studies…' : 'Search backtests…'
  const maxWidth = viewMode === 'grid' ? 1200 : 860

  return (
    <>
      {/* Navbar */}
      <nav className="navbar btk-navbar sticky-top px-3">
        <a className="btk-brand" href="/">
          bt<span className="btk-brand-dot">.</span>kit
        </a>
        <span className="btk-version">v2.0.0</span>
        <div className="ms-auto d-flex align-items-center gap-3">
          <Link
            to="/explore"
            style={{ fontSize: '0.8rem', color: '#93c5fd', textDecoration: 'none', fontWeight: 500 }}
          >
            <i className="bi bi-graph-up me-1" />
            Chart Explorer
          </Link>
          <span style={{ fontSize: '0.75rem', color: 'var(--btk-muted-dk)' }}>
            <i className="bi bi-hdd me-1" />
            es_options_backtests.db
          </span>
        </div>
      </nav>

      {/* Content */}
      <div className="container py-4" style={{ maxWidth }}>

        {/* Header */}
        <div className="d-flex align-items-center justify-content-between mb-4">
          <h5 className="mb-0 fw-semibold" style={{ color: '#e2e8f0' }}>Dashboard</h5>
        </div>

        {/* Toolbar: tabs · spacer · view toggle · search */}
        <div className="d-flex align-items-center gap-3 mb-3 flex-wrap">
          <ul className="nav btk-tabs" style={{ gap: 4 }}>
            <li className="nav-item">
              <button
                className={`nav-link ${tab === 'studies' ? 'active' : ''}`}
                onClick={() => { setTab('studies'); setActiveTagFilter(new Set()) }}
              >
                <i className="bi bi-grid-3x3-gap me-1" />
                Studies
                <span className="ms-2" style={{
                  fontSize: '0.7rem', background: 'rgba(255,255,255,0.08)',
                  borderRadius: 4, padding: '1px 6px', color: 'var(--btk-muted-dk)',
                }}>
                  {studies.length}
                </span>
              </button>
            </li>
            <li className="nav-item">
              <button
                className={`nav-link ${tab === 'backtests' ? 'active' : ''}`}
                onClick={() => { setTab('backtests'); setActiveTagFilter(new Set()) }}
              >
                <i className="bi bi-bar-chart-line me-1" />
                Backtests
                <span className="ms-2" style={{
                  fontSize: '0.7rem', background: 'rgba(255,255,255,0.08)',
                  borderRadius: 4, padding: '1px 6px', color: 'var(--btk-muted-dk)',
                }}>
                  {backtests.length}
                </span>
              </button>
            </li>
          </ul>

          <div className="flex-grow-1" />

          <ViewToggle mode={viewMode} onChange={setViewMode} />

          {/* Search */}
          <div className="input-group" style={{ maxWidth: 280, position: 'relative' }}>
            <span className="input-group-text btk-search-icon" style={{ borderRadius: '8px 0 0 8px' }}>
              <i className="bi bi-search" style={{ fontSize: '0.8rem' }} />
            </span>
            <input
              type="text"
              className="form-control btk-search"
              style={{ borderRadius: '0 8px 8px 0' }}
              placeholder={placeholder}
              value={query}
              onChange={e => setQuery(e.target.value)}
            />
            {query && (
              <button
                className="btn btn-link position-absolute"
                style={{ right: 8, top: '50%', transform: 'translateY(-50%)', zIndex: 5, color: 'var(--btk-muted-dk)', padding: 0 }}
                onClick={() => setQuery('')}
              >
                <i className="bi bi-x" />
              </button>
            )}
          </div>
        </div>

        {/* Tag filter bar */}
        {tagsInView.length > 0 && (
          <div className="mb-3">
            <TagFilterBar
              tags={tagsInView}
              active={activeTagFilter}
              onChange={setActiveTagFilter}
            />
          </div>
        )}

        {/* Result count */}
        {(query || activeTagFilter.size > 0) && (
          <p className="btk-summary mb-2">
            {items.length} result{items.length !== 1 ? 's' : ''}
            {query ? ` for "${query}"` : ''}
          </p>
        )}

        {/* States */}
        {loading && (
          <div className="btk-empty">
            <i className="bi bi-arrow-repeat" style={{ animation: 'spin 1.2s linear infinite', display: 'inline-block' }} />
            Loading…
          </div>
        )}

        {error && (
          <div className="btk-empty" style={{ color: '#f87171' }}>
            <i className="bi bi-exclamation-triangle" />
            {error}
          </div>
        )}

        {/* Content */}
        {!loading && !error && (
          items.length === 0 ? (
            <div className="btk-empty">
              <i className="bi bi-inbox" />
              {query || activeTagFilter.size > 0
                ? `No ${tab} match the current filters.`
                : `No ${tab} found.`}
            </div>
          ) : viewMode === 'grid' ? (
            tab === 'studies' ? (
              <BtkAgGrid
                rowData={filteredStudies}
                columnDefs={studyCols}
                onRowClicked={e => e.data && navigate(`/study/${e.data.id}`)}
                rowStyle={{ cursor: 'pointer' }}
                filterPlaceholder="Filter studies…"
                initialSortColId="created_at"
                initialSortDir="desc"
              />
            ) : (
              <>
                <div className="d-flex align-items-center mb-2" style={{ gap: 8 }}>
                  <CompositeScore
                    prefKey="composite.home"
                    onChange={setScoreConfig}
                  />
                </div>
                <BtkAgGrid
                  rowData={filteredBacktests}
                  columnDefs={finalBacktestCols}
                  onRowClicked={e => e.data && navigate(`/backtest/${e.data.id}`)}
                  rowStyle={{ cursor: 'pointer' }}
                  filterPlaceholder="Filter backtests…"
                  initialSortColId="created_at"
                  initialSortDir="desc"
                  prefKey="grid.home.backtests"
                />
              </>
            )
          ) : (
            <div className="btk-list">
              {tab === 'backtests'
                ? filteredBacktests.map(b => (
                    <BacktestRow key={b.id} b={b} onTagsChanged={updateBacktestTags} />
                  ))
                : filteredStudies.map(s => <StudyRow key={s.id} s={s} />)
              }
            </div>
          )
        )}
      </div>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </>
  )
}
