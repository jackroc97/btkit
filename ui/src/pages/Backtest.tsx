import { useState, useMemo, useEffect, useRef, useCallback } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { type ColDef } from 'ag-grid-community'
import PlotlyChart from '../components/charts/PlotlyChart'
import BtkAgGrid from '../components/BtkAgGrid'
import { TagPicker } from '../tags/TagPicker'
import { TagPillList } from '../tags/TagPill'
import type { Tag } from '../tags/TagsContext'

// ── Types ─────────────────────────────────────────────────────────────────────

interface ApiBacktest {
  id: number
  study_id: number | null
  combination_id: number
  strategy_name: string
  strategy_label: string
  initial_equity: number | null
  status: string
  created_at: string
  n_trades: number
  total_pnl: number
  avg_pnl: number | null
  win_rate: number | null
  avg_duration_min: number | null
  start_date: string | null
  end_date: string | null
  sharpe: number | null
  sortino: number | null
  calmar: number | null
  total_return_pct: number | null
  profit_factor: number | null
  max_drawdown: number | null
  max_drawdown_pct: number | null
  cagr: number | null
  avg_win: number | null
  avg_loss: number | null
  n_take_profit: number | null
  n_stop_loss: number | null
  tags: Tag[]
  params: Record<string, number | null>
}

interface ApiPosition {
  trade_num: number
  id:          number
  exit_reason: string
  open_date:   string
  exit_date:   string
  net_pnl:     number
  continuation_pnl: number | null
  continuation_exit_reason: string | null
  duration_min: number
  cum_pnl:     number
  return_pct:  number
}

interface BacktestSummary {
  id: number
  strategy_label: string
  combination_id: number
  status: string
}

interface CompareData {
  x: string[]
  y: number[]
  name: string
  color: string
  dash: 'dot' | 'dashdot' | 'solid'
}

type HistKey    = 'net_pnl' | 'return_pct' | 'duration_min'
type CompareMode = 'buyhold' | 'backtest' | 'livetrades'

// ── Seeded RNG ────────────────────────────────────────────────────────────────

function seededRng(seed: number) {
  let s = seed | 0
  return () => { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return (s >>> 0) / 4294967296 }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtDur(min: number) {
  if (min < 60) return `${min}m`
  const h = Math.floor(min / 60), m = min % 60
  return m ? `${h}h ${m}m` : `${h}h`
}

function computeCAGR(totalReturnPct: number | null, start: string | null, end: string | null): number | null {
  if (totalReturnPct === null || !start || !end) return null
  const years = (new Date(end).getTime() - new Date(start).getTime()) / (365.25 * 24 * 60 * 60 * 1000)
  if (years <= 0) return null
  return (Math.pow(1 + totalReturnPct / 100, 1 / years) - 1) * 100
}

function computeMaxDD(positions: ApiPosition[], initialEquity: number): number | null {
  if (positions.length === 0) return null
  const equity = [initialEquity, ...positions.map(p => initialEquity + p.cum_pnl)]
  let maxDD = 0, peak = equity[0]
  for (const e of equity) {
    if (e > peak) peak = e
    const dd = (e - peak) / peak * 100
    if (dd < maxDD) maxDD = dd
  }
  return maxDD
}

function computeFan(pnls: number[], initialEquity: number, nPaths = 120) {
  const rng = seededRng(99999)
  const n   = pnls.length
  const paths = Array.from({ length: nPaths }, () => {
    const arr = Array.from({ length: n }, () => pnls[Math.floor(rng() * n)])
    const eq = [initialEquity]
    for (const p of arr) eq.push(eq[eq.length - 1] + p)
    return eq
  })
  const pct = (vals: number[], p: number) =>
    [...vals].sort((a, b) => a - b)[Math.floor(vals.length * p)]
  const xs = Array.from({ length: n + 1 }, (_, i) => i)
  return {
    xs,
    p5:  xs.map(i => pct(paths.map(p => p[i]), 0.05)),
    p25: xs.map(i => pct(paths.map(p => p[i]), 0.25)),
    p50: xs.map(i => pct(paths.map(p => p[i]), 0.50)),
    p75: xs.map(i => pct(paths.map(p => p[i]), 0.75)),
    p95: xs.map(i => pct(paths.map(p => p[i]), 0.95)),
  }
}

function fmtVal(v: number | null, prefix = '', suffix = '', decimals = 2): string {
  if (v === null || v === undefined) return '—'
  return `${prefix}${v < 0 ? '-' : ''}${Math.abs(v).toFixed(decimals)}${suffix}`
}

function fmtParamTags(params: Record<string, number | null>): string[] {
  const parts: string[] = []
  if (params.delta != null) parts.push(`δ=${params.delta}`)
  if (params.dte != null) parts.push(`DTE=${Math.round(params.dte)}`)
  if (params.take_profit_pct != null) parts.push(`TP=${Math.round(params.take_profit_pct * 100)}%`)
  if (params.stop_loss != null) parts.push(`SL=${params.stop_loss}`)
  if (params.min_credit != null) parts.push(`min=$${params.min_credit}`)
  return parts
}

// ── Dark layout ───────────────────────────────────────────────────────────────

const DARK: Record<string, unknown> = {
  paper_bgcolor: '#161b27',
  plot_bgcolor:  '#161b27',
  font:          { color: '#94a3b8', size: 11, family: 'Inter, system-ui, sans-serif' },
  showlegend:    false,
  margin:        { t: 16, r: 16, b: 44, l: 72 },
  hoverlabel:    { bgcolor: '#1e2535', bordercolor: '#2a3245', font: { color: '#e2e8f0', size: 11 } },
  xaxis: { gridcolor: '#2a3245', linecolor: '#2a3245', tickcolor: '#64748b', zeroline: false },
  yaxis: { gridcolor: '#2a3245', linecolor: '#2a3245', tickcolor: '#64748b', zeroline: false },
}

// ── Histogram options ─────────────────────────────────────────────────────────

const HIST_OPTIONS: { key: HistKey; label: string }[] = [
  { key: 'net_pnl',      label: 'Trade P&L ($)'      },
  { key: 'return_pct',   label: 'Trade Return (%)'   },
  { key: 'duration_min', label: 'Hold Duration (min)' },
]

// ── Input style ───────────────────────────────────────────────────────────────

const INPUT_STYLE: React.CSSProperties = {
  background: '#1e2535', border: '1px solid #2a3245', borderRadius: 6,
  color: '#e2e8f0', fontSize: '0.8rem', padding: '5px 10px', outline: 'none',
}

const LBL: React.CSSProperties = {
  fontSize: '0.68rem', fontWeight: 600, letterSpacing: '0.05em',
  textTransform: 'uppercase', color: '#64748b', display: 'block', marginBottom: 4,
}

// ── Trade table ───────────────────────────────────────────────────────────────

function TradeTable({ positions, backtestId }: { positions: ApiPosition[]; backtestId: string | undefined }) {
  const navigate = useNavigate()

  const columnDefs: ColDef<ApiPosition>[] = [
    {
      field: 'trade_num', headerName: '#', width: 64,
      cellStyle: { color: 'var(--btk-muted-dk)' },
    },
    { field: 'open_date',  headerName: 'Open',  minWidth: 90 },
    {
      field: 'exit_date', headerName: 'Close', minWidth: 90,
      cellStyle: { color: 'var(--btk-muted-dk)' },
    },
    {
      field: 'duration_min', headerName: 'Duration', width: 90,
      valueFormatter: p => fmtDur(p.value as number),
      cellStyle: { color: 'var(--btk-muted-dk)' },
    },
    {
      field: 'exit_reason', headerName: 'Exit', width: 84,
      cellRenderer: (p: { value: string }) => (
        <span className={`btk-exit-badge ${p.value.toLowerCase()}`}>{p.value}</span>
      ),
    },
    {
      field: 'net_pnl', headerName: 'Spread P&L', width: 104,
      valueFormatter: p => { const v = p.value as number; return `${v >= 0 ? '+' : ''}${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}` },
      cellStyle: p => ({ color: (p.value as number) >= 0 ? '#4ade80' : '#f87171', fontWeight: 600 }),
    },
    {
      field: 'continuation_pnl', headerName: 'Cont. P&L', width: 104,
      valueFormatter: p => {
        if (p.value == null) return '—'
        const v = p.value as number
        return `${v >= 0 ? '+' : ''}${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
      },
      cellStyle: p => ({ color: p.value == null ? '#475569' : (p.value as number) >= 0 ? '#4ade80' : '#f87171' }),
    },
    {
      field: 'return_pct', headerName: 'Return', width: 90,
      valueFormatter: p => { const v = p.value as number; return `${v >= 0 ? '+' : ''}${v.toFixed(2)}%` },
      cellStyle: p => ({ color: (p.value as number) >= 0 ? '#4ade80' : '#f87171' }),
    },
    {
      field: 'cum_pnl', headerName: 'Cum. P&L', width: 104,
      valueFormatter: p => { const v = p.value as number; return `${v >= 0 ? '+' : ''}${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}` },
      cellStyle: p => ({ color: (p.value as number) >= 0 ? '#e2e8f0' : '#f87171' }),
    },
  ]

  return (
    <BtkAgGrid
      rowData={positions}
      columnDefs={columnDefs}
      onRowClicked={e => e.data && navigate(`/backtest/${backtestId}/trade/${e.data.id}`)}
      rowStyle={{ cursor: 'pointer' }}
      filterPlaceholder="Filter trades…"
      initialSortColId="trade_num"
      initialSortDir="asc"
    />
  )
}

// ── Compare panel ─────────────────────────────────────────────────────────────

function ComparePanel({
  bt, compareData, onLoad, onClear,
}: {
  bt: ApiBacktest
  compareData: CompareData | null
  onLoad: (d: CompareData) => void
  onClear: () => void
}) {
  const [mode, setMode]           = useState<CompareMode>('buyhold')
  const [ticker, setTicker]       = useState('SPY')
  const [qty, setQty]             = useState(1)
  const [btId, setBtId]           = useState<number | ''>('')
  const [backtests, setBacktests] = useState<BacktestSummary[]>([])
  const [csvFile, setCsvFile]     = useState<File | null>(null)
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState<string | null>(null)
  const fileRef                   = useRef<HTMLInputElement>(null)

  // Fetch backtest list when mode switches to backtest
  useEffect(() => {
    if (mode !== 'backtest' || backtests.length > 0) return
    fetch('/api/backtests').then(r => r.json()).then((rows: BacktestSummary[]) =>
      setBacktests(rows.filter(b => b.id !== bt.id && b.status === 'completed'))
    ).catch(() => {})
  }, [mode, backtests.length, bt.id])

  async function load() {
    setLoading(true)
    setError(null)
    try {
      if (mode === 'buyhold') {
        const qs = new URLSearchParams({ ticker: ticker.trim(), qty: String(qty) })
        if (bt.start_date) qs.set('start', bt.start_date)
        if (bt.end_date)   qs.set('end',   bt.end_date)
        const r = await fetch(`/api/compare/buyhold?${qs}`)
        if (!r.ok) { const e = await r.json(); throw new Error(e.detail ?? `HTTP ${r.status}`) }
        const d = await r.json()
        onLoad({ ...d, color: '#fbbf24', dash: 'dot' })

      } else if (mode === 'backtest') {
        if (!btId) return
        const r = await fetch(`/api/backtests/${btId}/positions`)
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        const pos: ApiPosition[] = await r.json()
        if (pos.length === 0) throw new Error('No positions in selected backtest')
        const found = backtests.find(b => b.id === btId)
        const label = found ? `#${btId} ${found.strategy_label}` : `Backtest #${btId}`
        onLoad({
          x: [pos[0].open_date, ...pos.map(p => p.exit_date)],
          y: [0, ...pos.map(p => p.cum_pnl)],
          name: label,
          color: '#2dd4bf',
          dash: 'dashdot',
        })

      } else {
        if (!csvFile) return
        const form = new FormData()
        form.append('file', csvFile)
        const qs = new URLSearchParams()
        if (bt.start_date) qs.set('start', bt.start_date)
        if (bt.end_date)   qs.set('end',   bt.end_date)
        const r = await fetch(`/api/compare/livetrades?${qs}`, { method: 'POST', body: form })
        if (!r.ok) { const e = await r.json(); throw new Error(e.detail ?? `HTTP ${r.status}`) }
        const d = await r.json()
        onLoad({ ...d, color: '#a78bfa', dash: 'solid' })
      }
    } catch (e) {
      setError(String(e).replace('Error: ', ''))
    } finally {
      setLoading(false)
    }
  }

  const modeBtns: { key: CompareMode; label: string }[] = [
    { key: 'buyhold',    label: 'Buy & Hold' },
    { key: 'backtest',   label: 'Backtest'   },
    { key: 'livetrades', label: 'Live Trades (CSV)' },
  ]

  const canLoad = mode === 'buyhold'
    ? ticker.trim().length > 0
    : mode === 'backtest'
      ? !!btId
      : !!csvFile

  return (
    <div className="btk-chart-card mb-3" style={{ padding: '16px 20px' }}>
      {/* Header */}
      <div className="d-flex align-items-center justify-content-between mb-3">
        <span className="btk-chart-title">Compare overlay</span>
        {compareData && (
          <div className="d-flex align-items-center gap-2">
            <span style={{ fontSize: '0.72rem', color: compareData.color }}>
              ● {compareData.name}
            </span>
            <button onClick={onClear} style={{
              background: 'transparent', border: '1px solid #2a3245', borderRadius: 4,
              color: '#64748b', cursor: 'pointer', fontSize: '0.72rem', padding: '2px 8px',
            }}>
              Clear
            </button>
          </div>
        )}
      </div>

      {/* Mode selector */}
      <div className="d-flex gap-2 mb-3" style={{ flexWrap: 'wrap' }}>
        {modeBtns.map(({ key, label }) => (
          <button key={key} onClick={() => setMode(key)} style={{
            background: mode === key ? '#2563eb' : '#1e2535',
            border: `1px solid ${mode === key ? '#2563eb' : '#2a3245'}`,
            borderRadius: 6, color: mode === key ? '#fff' : '#94a3b8',
            cursor: 'pointer', fontSize: '0.78rem', fontWeight: 500,
            padding: '5px 14px', transition: 'all 0.15s',
          }}>
            {label}
          </button>
        ))}
      </div>

      {/* Mode inputs */}
      <div className="d-flex align-items-end gap-2" style={{ flexWrap: 'wrap' }}>
        {mode === 'buyhold' && (
          <>
            <div>
              <label style={LBL}>Ticker</label>
              <input
                style={{ ...INPUT_STYLE, width: 90, textTransform: 'uppercase' }}
                value={ticker}
                onChange={e => setTicker(e.target.value.toUpperCase())}
                onKeyDown={e => e.key === 'Enter' && canLoad && load()}
                placeholder="SPY"
              />
            </div>
            <div>
              <label style={LBL}>Quantity</label>
              <input
                type="number" min={1} step={1}
                style={{ ...INPUT_STYLE, width: 80 }}
                value={qty}
                onChange={e => setQty(Number(e.target.value))}
              />
            </div>
          </>
        )}

        {mode === 'backtest' && (
          <div style={{ flex: 1, minWidth: 220 }}>
            <label style={LBL}>Backtest</label>
            <select
              style={{ ...INPUT_STYLE, width: '100%' }}
              value={btId}
              onChange={e => setBtId(e.target.value ? Number(e.target.value) : '')}
            >
              <option value="">Select a completed backtest…</option>
              {backtests.map(b => (
                <option key={b.id} value={b.id}>
                  #{b.id} · {b.strategy_label}
                </option>
              ))}
            </select>
          </div>
        )}

        {mode === 'livetrades' && (
          <>
            <input
              ref={fileRef}
              type="file"
              accept=".csv,text/csv"
              style={{ display: 'none' }}
              onChange={e => setCsvFile(e.target.files?.[0] ?? null)}
            />
            <div>
              <label style={LBL}>CSV file</label>
              <div className="d-flex align-items-center gap-2">
                <button onClick={() => fileRef.current?.click()} style={{
                  ...INPUT_STYLE, cursor: 'pointer',
                  color: csvFile ? '#e2e8f0' : '#64748b',
                }}>
                  <i className="bi bi-file-earmark-text me-1" />
                  {csvFile ? csvFile.name : 'Choose file…'}
                </button>
                {csvFile && (
                  <button onClick={() => setCsvFile(null)} style={{
                    background: 'transparent', border: 'none', color: '#64748b', cursor: 'pointer', padding: 0,
                  }}>
                    <i className="bi bi-x" />
                  </button>
                )}
              </div>
            </div>
            <div style={{ fontSize: '0.72rem', color: '#475569', alignSelf: 'flex-end', marginBottom: 6, lineHeight: 1.4 }}>
              Requires columns: <code style={{ color: '#93c5fd' }}>date</code>, <code style={{ color: '#93c5fd' }}>pnl</code>
            </div>
          </>
        )}

        <button
          onClick={load}
          disabled={loading || !canLoad}
          style={{
            background: loading || !canLoad ? '#1e2535' : '#2563eb',
            border: '1px solid #2563eb', borderRadius: 6,
            color: loading || !canLoad ? '#475569' : '#fff',
            cursor: loading || !canLoad ? 'not-allowed' : 'pointer',
            fontSize: '0.8rem', fontWeight: 600, padding: '5px 18px',
            transition: 'all 0.15s', alignSelf: 'flex-end',
          }}
        >
          {loading ? <><i className="bi bi-arrow-repeat me-1" style={{ animation: 'spin 1.2s linear infinite', display: 'inline-block' }} />Loading…</> : 'Load'}
        </button>
      </div>

      {error && (
        <div style={{ marginTop: 10, color: '#f87171', fontSize: '0.78rem' }}>
          <i className="bi bi-exclamation-circle me-1" />{error}
        </div>
      )}
    </div>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function Backtest() {
  const { id }                    = useParams<{ id: string }>()
  const [bt, setBt]               = useState<ApiBacktest | null>(null)
  const [positions, setPositions] = useState<ApiPosition[]>([])
  const [loading, setLoading]     = useState(true)
  const [error, setError]         = useState<string | null>(null)
  const [histKey, setHistKey]     = useState<HistKey>('net_pnl')
  const [xMode, setXMode]         = useState<'date' | 'trade'>('date')
  const [showCompare, setShowCompare] = useState(false)
  const [compareData, setCompareData] = useState<CompareData | null>(null)
  const [localTags, setLocalTags] = useState<Tag[]>([])

  const handleTagsChanged = useCallback((tags: Tag[]) => {
    setLocalTags(tags)
    setBt(prev => prev ? { ...prev, tags } : prev)
  }, [])

  useEffect(() => {
    if (!id) return
    setCompareData(null)
    Promise.all([
      fetch(`/api/backtests/${id}`).then(r => { if (!r.ok) throw new Error(`${r.status}`); return r.json() }),
      fetch(`/api/backtests/${id}/positions`).then(r => { if (!r.ok) throw new Error(`${r.status}`); return r.json() }),
    ])
      .then(([b, p]) => { setBt(b); setPositions(p); setLocalTags(b.tags ?? []) })
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false))
  }, [id])

  const initial = bt?.initial_equity ?? 100_000

  // Computed metrics
  const cagr           = useMemo(() => computeCAGR(bt?.total_return_pct ?? null, bt?.start_date ?? null, bt?.end_date ?? null), [bt])
  const maxDD          = useMemo(() => computeMaxDD(positions, initial), [positions, initial])
  const recoveryFactor = useMemo(() => {
    if (maxDD == null || maxDD >= 0 || bt == null) return null
    const dollarDD = Math.abs(maxDD / 100) * initial
    return dollarDD > 0 ? bt.total_pnl / dollarDD : null
  }, [maxDD, bt, initial])

  // Equity series
  const equity   = useMemo(() => [0, ...positions.map(p => p.cum_pnl)], [positions])
  const tradeXs  = useMemo(() => Array.from({ length: equity.length }, (_, i) => i), [equity])
  const dateXs   = useMemo(() => {
    if (!bt?.start_date || positions.length === 0) return []
    return [bt.start_date, ...positions.map(p => p.exit_date)]
  }, [bt, positions])

  // Bootstrap fan (always trade#)
  const fan = useMemo(() => computeFan(positions.map(p => p.net_pnl), 0), [positions])

  // Equity traces — include compare overlay when in date mode
  const equityTraces = useMemo(() => {
    const xs = xMode === 'date' && dateXs.length > 0 ? dateXs : tradeXs
    const hasCompare = !!compareData && xMode === 'date'
    const traces: unknown[] = [
      ...(xMode !== 'date' ? [{
        x: tradeXs, y: Array(tradeXs.length).fill(0),
        type: 'scatter', mode: 'lines',
        line: { width: 1, color: 'rgba(100,116,139,0.2)', dash: 'dot' }, hoverinfo: 'skip',
      }] : []),
      {
        x: xs, y: equity,
        name: bt?.strategy_label ?? 'Backtest',
        type: 'scatter', mode: 'lines',
        line: { width: 2, color: '#2563EB' },
        fill: 'tozeroy', fillcolor: 'rgba(37,99,235,0.06)',
        hovertemplate: xMode === 'date' ? '%{x}<br>$%{y:,.0f}<extra></extra>' : 'Trade #%{x}<br>$%{y:,.0f}<extra></extra>',
      },
    ]
    if (hasCompare) {
      traces.push({
        x: compareData.x, y: compareData.y,
        name: compareData.name,
        type: 'scatter', mode: 'lines',
        line: { width: 1.5, color: compareData.color, dash: compareData.dash },
        hovertemplate: `${compareData.name}<br>%{x}<br>$%{y:+,.0f}<extra></extra>`,
      })
    }
    return traces
  }, [tradeXs, equity, xMode, dateXs, compareData, bt])

  // Fan traces
  const fanTraces = useMemo(() => [
    { x: fan.xs, y: fan.p5,  type: 'scatter', mode: 'lines', line: { color: 'transparent', width: 0 }, hoverinfo: 'skip' },
    { x: fan.xs, y: fan.p95, type: 'scatter', mode: 'lines', line: { color: 'transparent', width: 0 }, fill: 'tonexty', fillcolor: 'rgba(37,99,235,0.07)', hoverinfo: 'skip' },
    { x: fan.xs, y: fan.p25, type: 'scatter', mode: 'lines', line: { color: 'transparent', width: 0 }, hoverinfo: 'skip' },
    { x: fan.xs, y: fan.p75, type: 'scatter', mode: 'lines', line: { color: 'transparent', width: 0 }, fill: 'tonexty', fillcolor: 'rgba(37,99,235,0.14)', hoverinfo: 'skip' },
    { x: fan.xs, y: fan.p50, type: 'scatter', mode: 'lines', line: { color: 'rgba(37,99,235,0.55)', width: 1.5, dash: 'dot' }, hovertemplate: 'Trade #%{x}<br>Median $%{y:,.0f}<extra></extra>' },
    { x: tradeXs, y: equity, type: 'scatter', mode: 'lines', line: { color: '#2563EB', width: 2.5 }, hovertemplate: 'Trade #%{x}<br>Actual $%{y:,.0f}<extra></extra>' },
  ], [fan, tradeXs, equity])

  // Histogram traces
  const histTraces = useMemo(() => {
    const isBicolor = histKey !== 'duration_min'
    if (isBicolor) {
      const wins   = positions.filter(p => p.net_pnl >= 0).map(p => p[histKey])
      const losses = positions.filter(p => p.net_pnl <  0).map(p => p[histKey])
      return [
        { x: wins,   type: 'histogram', name: 'Win',  marker: { color: 'rgba(74,222,128,0.55)',  line: { color: '#4ade80', width: 1 } }, autobinx: true, opacity: 0.85 },
        { x: losses, type: 'histogram', name: 'Loss', marker: { color: 'rgba(248,113,113,0.55)', line: { color: '#f87171', width: 1 } }, autobinx: true, opacity: 0.85 },
      ]
    }
    return [{ x: positions.map(p => p[histKey]), type: 'histogram', marker: { color: 'rgba(37,99,235,0.65)', line: { color: '#2563EB', width: 1 } }, autobinx: true }]
  }, [positions, histKey])

  const histLayout = useMemo(() => ({
    ...DARK, barmode: 'overlay', showlegend: false,
    xaxis: { ...(DARK.xaxis as object), title: { text: HIST_OPTIONS.find(o => o.key === histKey)?.label, font: { size: 10 } } },
    yaxis: { ...(DARK.yaxis as object), title: { text: 'Trades', font: { size: 10 } } },
    bargap: 0.04,
  }), [histKey])

  const equityLayout = useMemo(() => ({
    ...DARK,
    showlegend: !!compareData && xMode === 'date',
    legend: { font: { color: '#94a3b8', size: 10 }, bgcolor: 'transparent', x: 0.01, y: 0.99 },
    xaxis: xMode === 'date'
      ? { ...(DARK.xaxis as object), type: 'date', title: { text: 'Date', font: { size: 10 } } }
      : { ...(DARK.xaxis as object), title: { text: 'Trade #', font: { size: 10 } } },
    yaxis: { ...(DARK.yaxis as object), tickformat: '$,.0f', title: { text: 'Cumulative P&L', font: { size: 10 } } },
    hovermode: 'x unified',
  }), [xMode, compareData])

  const fanLayout = {
    ...DARK,
    xaxis: { ...(DARK.xaxis as object), title: { text: 'Trade #', font: { size: 10 } } },
    yaxis: { ...(DARK.yaxis as object), tickformat: '$,.0f', title: { text: 'Cumulative P&L', font: { size: 10 } } },
    hovermode: 'x unified',
  }

  // Metrics grid
  const winPct = bt?.win_rate != null ? bt.win_rate * 100 : null
  const avgDur = bt?.avg_duration_min != null ? Math.round(bt.avg_duration_min) : null

  const metrics = bt ? [
    { label: 'Total Return',   value: fmtVal(bt.total_return_pct, (bt.total_return_pct ?? 0) >= 0 ? '+' : '', '%', 1), color: (bt.total_return_pct ?? 0) >= 0 ? 'positive' : 'negative' },
    { label: 'CAGR',           value: fmtVal(cagr, (cagr ?? 0) >= 0 ? '+' : '', '%', 1),                              color: (cagr ?? 0) >= 0 ? 'positive' : 'negative'           },
    { label: 'Sharpe Ratio',   value: fmtVal(bt.sharpe),                                                               color: bt.sharpe != null ? (bt.sharpe >= 1.5 ? 'positive' : bt.sharpe >= 0.8 ? 'amber' : 'negative') : 'neutral' },
    { label: 'Sortino Ratio',  value: fmtVal(bt.sortino),                                                              color: bt.sortino != null ? (bt.sortino >= 1.5 ? 'positive' : bt.sortino >= 0.8 ? 'amber' : 'negative') : 'neutral' },
    { label: 'Calmar Ratio',   value: fmtVal(bt.calmar),                                                               color: bt.calmar == null ? 'neutral' : bt.calmar >= 1 ? 'positive' : bt.calmar >= 0.5 ? 'amber' : 'negative' },
    { label: 'Max Drawdown',   value: fmtVal(maxDD, '', '%', 1),                                                       color: 'negative' },
    { label: 'Recovery Factor', value: fmtVal(recoveryFactor),                                                          color: recoveryFactor == null ? 'neutral' : recoveryFactor >= 1 ? 'positive' : 'negative' },
    { label: 'Win Rate',       value: fmtVal(winPct, '', '%', 1),                                                      color: 'neutral'  },
    { label: 'Profit Factor',  value: fmtVal(bt.profit_factor),                                                        color: bt.profit_factor == null ? 'neutral' : bt.profit_factor >= 1 ? 'positive' : 'negative' },
    { label: 'Avg Trade P&L',  value: bt.avg_pnl != null ? `${bt.avg_pnl >= 0 ? '+' : ''}$${Math.abs(bt.avg_pnl).toFixed(0)}` : '—', color: (bt.avg_pnl ?? 0) >= 0 ? 'positive' : 'negative' },
    { label: 'Avg Win',        value: bt.avg_win  != null ? `+$${bt.avg_win.toFixed(0)}`              : '—',           color: 'positive' },
    { label: 'Avg Loss',       value: bt.avg_loss != null ? `-$${Math.abs(bt.avg_loss).toFixed(0)}`   : '—',           color: 'negative' },
    { label: 'Num Trades',     value: String(bt.n_trades),                                                             color: 'neutral'  },
    { label: 'Take Profit',    value: bt.n_take_profit != null ? String(bt.n_take_profit) : '—',                       color: 'neutral'  },
    { label: 'Stop Loss',      value: bt.n_stop_loss   != null ? String(bt.n_stop_loss)   : '—',                       color: 'neutral'  },
    { label: 'Avg Duration',   value: avgDur != null ? fmtDur(avgDur) : '—',                                          color: 'neutral'  },
    ...(bt.params.dte != null ? [{ label: 'Target DTE', value: String(Math.round(bt.params.dte)), color: 'neutral' as const }] : []),
  ] : []

  const paramTags = bt ? fmtParamTags(bt.params) : []

  return (
    <>
      <nav className="navbar btk-navbar sticky-top px-3">
        <a className="btk-brand" href="/">bt<span className="btk-brand-dot">.</span>kit</a>
        <span className="btk-version">v2.0.0</span>
        <div className="ms-auto d-flex align-items-center gap-2">
          <span style={{ fontSize: '0.75rem', color: 'var(--btk-muted-dk)' }}>
            <i className="bi bi-hdd me-1" />es_options_backtests.db
          </span>
        </div>
      </nav>

      <div className="container py-4" style={{ maxWidth: 1100 }}>

        <nav aria-label="breadcrumb" className="btk-breadcrumb mb-3">
          <ol className="breadcrumb mb-0">
            <li className="breadcrumb-item"><Link to="/">Dashboard</Link></li>
            {bt?.study_id && (
              <li className="breadcrumb-item"><Link to={`/study/${bt.study_id}`}>Study {bt.study_id}</Link></li>
            )}
            <li className="breadcrumb-item active" aria-current="page">
              {loading ? 'Loading…' : (bt ? `${bt.strategy_label} #${bt.combination_id}` : `Backtest ${id}`)}
            </li>
          </ol>
        </nav>

        {loading && (
          <div className="btk-empty">
            <i className="bi bi-arrow-repeat" style={{ animation: 'spin 1.2s linear infinite', display: 'inline-block' }} />
            Loading…
          </div>
        )}

        {error && (
          <div className="btk-empty" style={{ color: '#f87171' }}>
            <i className="bi bi-exclamation-triangle" />{error}
          </div>
        )}

        {!loading && !error && bt && (
          <>
            {/* Header */}
            <div className="d-flex align-items-start justify-content-between gap-3 mb-3">
              <div>
                <div className="d-flex align-items-center gap-2 flex-wrap mb-1">
                  <h5 className="mb-0 fw-semibold" style={{ color: '#e2e8f0' }}>{bt.strategy_label}</h5>
                  {localTags.length > 0 && <TagPillList tags={localTags} maxVisible={5} size="md" />}
                  <TagPicker
                    backtestId={bt.id}
                    currentTags={localTags}
                    onChanged={handleTagsChanged}
                  />
                </div>
                <div className="btk-item-sub">
                  <span style={{ color: '#60a5fa', fontFamily: 'monospace', fontSize: '0.75rem' }}>{bt.strategy_name}</span>
                  {bt.start_date && bt.end_date && (
                    <><span className="mx-2">·</span>
                    <span>{new Date(bt.start_date).toLocaleDateString('en-US', { month: 'short', year: 'numeric' })} – {new Date(bt.end_date).toLocaleDateString('en-US', { month: 'short', year: 'numeric' })}</span></>
                  )}
                  {paramTags.map((t, i) => <span key={i} className="btk-param-tag ms-1">{t}</span>)}
                </div>
              </div>
              <span className={`btk-badge ${bt.status}`}>{bt.status}</span>
            </div>

            {/* Metrics grid */}
            <div className="row g-2 mb-3">
              {metrics.map(m => (
                <div className="col-6 col-sm-4 col-lg-2" key={m.label} style={{ minWidth: 120 }}>
                  <div className="btk-metric-card">
                    <div className="btk-metric-label">{m.label}</div>
                    <div className={`btk-metric-value ${m.color}`}>{m.value}</div>
                  </div>
                </div>
              ))}
            </div>

            {/* Charts row */}
            <div className="row g-3 mb-3">
              <div className="col-lg-7">
                <div className="btk-chart-card h-100">
                  <div className="d-flex align-items-center justify-content-between px-3 pt-3 pb-1" style={{ flexWrap: 'wrap', gap: 8 }}>
                    <span className="btk-chart-title">Equity Curve</span>
                    <div className="d-flex align-items-center gap-2">
                      {/* Compare toggle */}
                      <button
                        onClick={() => setShowCompare(v => !v)}
                        title="Compare overlay"
                        style={{
                          background: showCompare ? 'rgba(37,99,235,0.2)' : 'transparent',
                          border: `1px solid ${showCompare ? '#2563eb' : '#2a3245'}`,
                          borderRadius: 5, color: showCompare ? '#93c5fd' : '#64748b',
                          cursor: 'pointer', fontSize: '0.72rem', fontWeight: 600,
                          padding: '2px 9px', transition: 'all 0.15s', letterSpacing: '0.02em',
                        }}
                      >
                        <i className="bi bi-plus-slash-minus me-1" style={{ fontSize: '0.75rem' }} />
                        Compare
                        {compareData && <span style={{ marginLeft: 5, width: 6, height: 6, borderRadius: '50%', background: compareData.color, display: 'inline-block', verticalAlign: 'middle' }} />}
                      </button>
                      {/* xMode toggle */}
                      <div className="d-flex" style={{ background: '#1e2535', borderRadius: 6, padding: 2, gap: 2 }}>
                        {(['date', 'trade'] as const).map(m => (
                          <button key={m} onClick={() => setXMode(m)} style={{
                            background: xMode === m ? '#2563eb' : 'transparent',
                            border: 'none', borderRadius: 4,
                            color: xMode === m ? '#fff' : '#64748b',
                            cursor: 'pointer', fontSize: '0.7rem', fontWeight: 600,
                            padding: '2px 10px', transition: 'background 0.15s, color 0.15s',
                          }}>
                            {m === 'date' ? 'Date' : 'Trade #'}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                  {positions.length > 0 ? (
                    <PlotlyChart data={equityTraces} layout={{ ...equityLayout, autosize: true }} style={{ width: '100%', height: 280 }} />
                  ) : (
                    <div className="btk-empty" style={{ height: 280 }}>
                      <i className="bi bi-info-circle" />No positions
                    </div>
                  )}
                </div>
              </div>

              <div className="col-lg-5">
                <div className="btk-chart-card h-100">
                  <div className="d-flex align-items-center justify-content-between px-3 pt-3 pb-1">
                    <span className="btk-chart-title">Distribution</span>
                    <select className="form-select btk-metric-select" value={histKey} onChange={e => setHistKey(e.target.value as HistKey)}>
                      {HIST_OPTIONS.map(o => <option key={o.key} value={o.key}>{o.label}</option>)}
                    </select>
                  </div>
                  {positions.length > 0 ? (
                    <PlotlyChart data={histTraces} layout={{ ...histLayout, autosize: true }} style={{ width: '100%', height: 280 }} />
                  ) : (
                    <div className="btk-empty" style={{ height: 280 }}>
                      <i className="bi bi-info-circle" />No positions
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Compare panel */}
            {showCompare && bt && (
              <ComparePanel
                bt={bt}
                compareData={compareData}
                onLoad={setCompareData}
                onClear={() => setCompareData(null)}
              />
            )}

            {/* Bootstrap equity fan */}
            {positions.length > 0 && (
              <div className="btk-chart-card mb-3">
                <div className="d-flex align-items-center justify-content-between px-3 pt-3 pb-1">
                  <span className="btk-chart-title">Bootstrap Equity Fan</span>
                  <span className="btk-item-sub" style={{ fontSize: '0.72rem' }}>
                    120 resampled paths · &nbsp;
                    <span style={{ display: 'inline-block', width: 20, height: 2, background: '#2563EB', verticalAlign: 'middle', marginRight: 4 }} />actual &nbsp;
                    <span style={{ display: 'inline-block', width: 12, height: 12, background: 'rgba(37,99,235,0.25)', borderRadius: 2, verticalAlign: 'middle', marginRight: 4 }} />P25–P75 &nbsp;
                    <span style={{ display: 'inline-block', width: 12, height: 12, background: 'rgba(37,99,235,0.10)', borderRadius: 2, verticalAlign: 'middle', marginRight: 4 }} />P5–P95
                  </span>
                </div>
                <PlotlyChart data={fanTraces} layout={{ ...fanLayout, autosize: true }} style={{ width: '100%', height: 300 }} />
              </div>
            )}

            {/* Trade table */}
            <div className="mb-2">
              <span className="btk-chart-title">Trades</span>
            </div>
            {positions.length > 0 ? (
              <TradeTable positions={positions} backtestId={id} />
            ) : (
              <div className="btk-empty"><i className="bi bi-inbox" />No trades found.</div>
            )}
          </>
        )}
      </div>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </>
  )
}
