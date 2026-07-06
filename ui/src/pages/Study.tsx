import { useState, useMemo, useRef, useEffect, useCallback } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { type ColDef, type ICellRendererParams, type ValueGetterParams } from 'ag-grid-community'
import PlotlyChart from '../components/charts/PlotlyChart'
import BtkAgGrid from '../components/BtkAgGrid'
import CompositeScore, { type ScoreConfig } from '../components/CompositeScore'
import { TagFilterBar } from '../tags/TagFilterBar'
import { TagPicker } from '../tags/TagPicker'
import { TagPillList } from '../tags/TagPill'
import type { Tag } from '../tags/TagsContext'

// ── Types ─────────────────────────────────────────────────────────────────────

type Status = 'completed' | 'running' | 'error'
type HeatMetric = 'sharpe' | 'sortino' | 'calmar' | 'total_return_pct' | 'win_rate_pct' | 'profit_factor' | 'avg_pnl' | 'recovery_factor'

interface ApiBacktest {
  id: number
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
  recovery_factor: number | null
  mean_pnl_ci_lower: number | null
  mean_pnl_ci_upper: number | null
  win_rate_ci_lower: number | null
  win_rate_ci_upper: number | null
  tags: Tag[]
  params: Record<string, number | null>
}

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
  backtests: ApiBacktest[]
}

interface EquityCurve {
  backtest_id: number
  initial_equity: number
  cum_pnl: number[]
  exit_dates: string[]
}

interface HeatmapConfig {
  id:        number
  xAxis:     string
  yAxis:     string
  metric:    HeatMetric
  constants: Record<string, string>
  strategy:  string | null
}

interface DailyPnl {
  trade_date:  string
  total_pnl:   number
  n_backtests: number
  n_trades:    number
}

// ── Metric definitions ────────────────────────────────────────────────────────

interface HeatMetricDef {
  key:          HeatMetric
  label:        string
  higherBetter: boolean
  fmt:          (v: number) => string
}

const HEAT_METRICS: HeatMetricDef[] = [
  { key: 'sharpe',           label: 'Sharpe Ratio',     higherBetter: true,  fmt: v => v.toFixed(2)                           },
  { key: 'sortino',          label: 'Sortino Ratio',    higherBetter: true,  fmt: v => v.toFixed(2)                           },
  { key: 'calmar',           label: 'Calmar Ratio',     higherBetter: true,  fmt: v => v.toFixed(2)                           },
  { key: 'total_return_pct', label: 'Total Return %',   higherBetter: true,  fmt: v => `${v > 0 ? '+' : ''}${v.toFixed(1)}%`  },
  { key: 'win_rate_pct',     label: 'Win Rate %',       higherBetter: true,  fmt: v => `${v.toFixed(1)}%`                      },
  { key: 'profit_factor',    label: 'Profit Factor',    higherBetter: true,  fmt: v => v.toFixed(2)                            },
  { key: 'recovery_factor',  label: 'Recovery Factor',  higherBetter: true,  fmt: v => v.toFixed(2)                            },
  { key: 'avg_pnl',          label: 'Avg P&L ($)',      higherBetter: true,  fmt: v => `$${v.toFixed(0)}`                      },
]

const HIST_METRICS = [
  { key: 'sharpe',           label: 'Sharpe Ratio',     tickformat: '.2f'  },
  { key: 'sortino',          label: 'Sortino Ratio',    tickformat: '.2f'  },
  { key: 'calmar',           label: 'Calmar Ratio',     tickformat: '.2f'  },
  { key: 'total_return_pct', label: 'Total Return (%)', tickformat: '.1f'  },
  { key: 'win_rate_pct',     label: 'Win Rate (%)',     tickformat: '.1f'  },
  { key: 'profit_factor',    label: 'Profit Factor',    tickformat: '.2f'  },
  { key: 'recovery_factor',  label: 'Recovery Factor',  tickformat: '.2f'  },
  { key: 'avg_pnl',          label: 'Avg P&L ($)',      tickformat: ',.0f' },
  { key: 'n_trades',         label: 'Num Trades',       tickformat: 'd'    },
]

// Plotly colorscales
const SCALE_GOOD = [
  [0.00, 'rgba(185,28,28,0.90)'],
  [0.30, 'rgba(220,38,38,0.82)'],
  [0.50, 'rgba(161,98,7,0.85)'],
  [0.70, 'rgba(21,128,61,0.85)'],
  [1.00, 'rgba(22,163,74,0.92)'],
]
const SCALE_BAD = [...SCALE_GOOD].reverse().map(([t, c]) => [1 - (t as number), c])

// ── Dark layout ───────────────────────────────────────────────────────────────

const DARK_LAYOUT = {
  paper_bgcolor: '#161b27',
  plot_bgcolor:  '#161b27',
  font:   { color: '#94a3b8', size: 11, family: 'Inter, system-ui, sans-serif' },
  xaxis:  { gridcolor: '#2a3245', linecolor: '#2a3245', tickcolor: '#64748b', zeroline: false },
  yaxis:  { gridcolor: '#2a3245', linecolor: '#2a3245', tickcolor: '#64748b', zeroline: false },
  showlegend: false,
  margin: { t: 16, r: 16, b: 44, l: 72 },
  hoverlabel: { bgcolor: '#1e2535', bordercolor: '#2a3245', font: { color: '#e2e8f0', size: 11 } },
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtDate(iso: string | null): string {
  if (!iso) return '—'
  const d = new Date(iso)
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: '2-digit' })
}

function sharpeColor(v: number): string {
  return v >= 1.5 ? 'positive' : v >= 0.8 ? 'amber' : 'negative'
}

function fmt(v: number | null, prefix = '', suffix = '', decimals = 2): string {
  if (v === null || v === undefined) return '—'
  const s = Math.abs(v).toFixed(decimals)
  return `${prefix}${v < 0 ? '-' : ''}${s}${suffix}`
}

function valEq(a: number | null | undefined, b: string): boolean {
  if (a === null || a === undefined) return b === 'null' || b === ''
  return Math.abs(a - Number(b)) < 1e-9
}

function getAxisValues(backtests: ApiBacktest[], axis: string): string[] {
  const vals = new Set<string>()
  for (const b of backtests) {
    const v = b.params[axis]
    vals.add(v === null || v === undefined ? 'null' : String(v))
  }
  return [...vals].sort((a, z) => {
    const na = Number(a), nb = Number(z)
    if (!isNaN(na) && !isNaN(nb)) return na - nb
    if (a === 'null') return 1
    if (z === 'null') return -1
    return a.localeCompare(z)
  })
}

function getMetricValue(b: ApiBacktest, metric: HeatMetric): number | null {
  switch (metric) {
    case 'sharpe':           return b.sharpe
    case 'sortino':          return b.sortino
    case 'calmar':           return b.calmar
    case 'total_return_pct': return b.total_return_pct
    case 'win_rate_pct':     return b.win_rate != null ? b.win_rate * 100 : null
    case 'profit_factor':    return b.profit_factor
    case 'recovery_factor':  return b.recovery_factor
    case 'avg_pnl':          return b.avg_pnl
  }
}

function getHistValue(b: ApiBacktest, key: string): number | null {
  switch (key) {
    case 'sharpe':           return b.sharpe
    case 'sortino':          return b.sortino
    case 'calmar':           return b.calmar
    case 'total_return_pct': return b.total_return_pct
    case 'win_rate_pct':     return b.win_rate != null ? b.win_rate * 100 : null
    case 'profit_factor':    return b.profit_factor
    case 'recovery_factor':  return b.recovery_factor
    case 'avg_pnl':          return b.avg_pnl
    case 'n_trades':         return b.n_trades
    default:                 return null
  }
}

// ── Sub-components ────────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: Status }) {
  const label = status === 'running' ? (
    <><i className="bi bi-arrow-repeat me-1" style={{ animation: 'spin 1.2s linear infinite', display: 'inline-block' }} />running</>
  ) : status
  return <span className={`btk-badge ${status}`}>{label}</span>
}

// ── Backtests AG Grid ─────────────────────────────────────────────────────────

function BacktestTable({
  backtests,
  onTagsChanged,
  scoreConfig,
  studyId,
}: {
  backtests: ApiBacktest[]
  onTagsChanged: (backtestId: number, tags: Tag[]) => void
  scoreConfig: ScoreConfig | null
  studyId: string | undefined
}) {
  const navigate = useNavigate()

  const baseColumnDefs: ColDef<ApiBacktest>[] = [
    {
      field: 'combination_id', headerName: '#', width: 64,
      cellStyle: { color: 'var(--btk-muted-dk)' },
    },
    {
      field: 'strategy_label', headerName: 'Strategy', minWidth: 140, flex: 1,
      cellStyle: { color: '#60a5fa', fontFamily: 'monospace', fontSize: '0.78rem' },
    },
    {
      headerName: 'Tags', width: 170,
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
      headerName: 'δ', width: 72,
      valueGetter: p => p.data?.params?.delta,
      valueFormatter: p => p.value != null ? Number(p.value).toFixed(2) : '—',
    },
    {
      headerName: 'TP', width: 72,
      valueGetter: p => p.data?.params?.take_profit_pct,
      valueFormatter: p => p.value != null ? `${Math.round(Number(p.value) * 100)}%` : '—',
    },
    {
      headerName: 'SL', width: 72,
      valueGetter: p => p.data?.params?.stop_loss,
      valueFormatter: p => p.value != null ? String(p.value) : '—',
    },
    {
      headerName: 'DTE', width: 68,
      valueGetter: p => p.data?.params?.dte,
      valueFormatter: p => p.value != null ? String(Math.round(Number(p.value))) : '—',
    },
    {
      field: 'status', headerName: 'Status', width: 104,
      cellRenderer: (p: { value: Status }) => {
        const s = p.value
        if (s === 'running') return (
          <span className="btk-badge running">
            <i className="bi bi-arrow-repeat me-1" style={{ animation: 'spin 1.2s linear infinite', display: 'inline-block' }} />
            running
          </span>
        )
        return <span className={`btk-badge ${s}`}>{s}</span>
      },
    },
    {
      field: 'sharpe', headerName: 'Sharpe', width: 90,
      valueFormatter: p => p.value != null ? (p.value as number).toFixed(2) : '—',
      cellStyle: p => {
        const v = p.value as number | null
        if (v == null) return null
        return { color: v >= 1.5 ? '#4ade80' : v >= 0.8 ? '#f59e0b' : '#f87171', fontWeight: 600, fontSize: '0.825rem' }
      },
    },
    {
      field: 'sortino', headerName: 'Sortino', width: 90,
      valueFormatter: p => p.value != null ? (p.value as number).toFixed(2) : '—',
      cellStyle: p => {
        const v = p.value as number | null
        if (v == null) return null
        return { color: v >= 1.5 ? '#4ade80' : v >= 0.8 ? '#f59e0b' : '#f87171', fontWeight: 600, fontSize: '0.825rem' }
      },
    },
    {
      field: 'calmar', headerName: 'Calmar', width: 90,
      valueFormatter: p => p.value != null ? (p.value as number).toFixed(2) : '—',
      cellStyle: p => {
        const v = p.value as number | null
        if (v == null) return null
        return { color: v >= 1 ? '#4ade80' : v >= 0.5 ? '#f59e0b' : '#f87171', fontSize: '0.825rem' }
      },
    },
    {
      field: 'total_return_pct', headerName: 'Return', width: 90,
      valueFormatter: p => { const v = p.value as number | null; return v == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(1)}%` },
      cellStyle: p => ({ color: ((p.value as number | null) ?? 0) >= 0 ? '#4ade80' : '#f87171', fontSize: '0.825rem' }),
    },
    {
      field: 'win_rate', headerName: 'Win Rate', width: 90,
      valueFormatter: p => p.value != null ? `${((p.value as number) * 100).toFixed(1)}%` : '—',
    },
    {
      field: 'n_trades', headerName: 'Trades', width: 80,
    },
    {
      headerName: 'Final Equity', width: 120,
      valueGetter: p => {
        const init = p.data?.initial_equity
        const pnl  = p.data?.total_pnl ?? 0
        return init != null ? init + pnl : null
      },
      valueFormatter: p => { const v = p.value as number | null; return v == null ? '—' : `$${v.toLocaleString('en-US', { maximumFractionDigits: 0 })}` },
      cellStyle: p => {
        const init = (p.node?.data as ApiBacktest | undefined)?.initial_equity
        const v = p.value as number | null
        if (v == null || init == null) return null
        return { color: v >= init ? '#4ade80' : '#f87171' }
      },
    },
    {
      field: 'avg_pnl', headerName: 'Avg P&L', width: 100,
      valueFormatter: p => { const v = p.value as number | null; return v == null ? '—' : `${v >= 0 ? '+' : ''}$${Math.abs(v).toFixed(0)}` },
      cellStyle: p => ({ color: ((p.value as number | null) ?? 0) >= 0 ? '#4ade80' : '#f87171' }),
    },
    {
      field: 'profit_factor', headerName: 'Prof. Factor', width: 104,
      valueFormatter: p => p.value != null ? (p.value as number).toFixed(2) : '—',
    },
    {
      field: 'max_drawdown_pct', headerName: 'Max DD', width: 88,
      valueFormatter: p => p.value != null ? `-${(p.value as number).toFixed(1)}%` : '—',
      cellStyle: p => p.value != null ? { color: '#f87171', fontSize: '0.825rem' } : null,
    },
    {
      field: 'cagr', headerName: 'CAGR', width: 84,
      valueFormatter: p => { const v = p.value as number | null; return v == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(1)}%` },
      cellStyle: p => ({ color: ((p.value as number | null) ?? 0) >= 0 ? '#4ade80' : '#f87171', fontSize: '0.825rem' }),
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
      headerName: 'Mean P&L CI', width: 148,
      valueGetter: p => {
        const lo = p.data?.mean_pnl_ci_lower, hi = p.data?.mean_pnl_ci_upper
        return lo != null && hi != null ? (lo + hi) / 2 : null
      },
      cellRenderer: (p: { data?: ApiBacktest }) => {
        const lo = p.data?.mean_pnl_ci_lower, hi = p.data?.mean_pnl_ci_upper
        if (lo == null || hi == null) return '—'
        const fmtV = (v: number) => `${v >= 0 ? '+' : ''}$${Math.abs(v).toFixed(0)}`
        return <span style={{ fontSize: '0.78rem', color: '#94a3b8' }}>{fmtV(lo)} – {fmtV(hi)}</span>
      },
    },
    {
      headerName: 'Win Rate CI', width: 136,
      valueGetter: p => {
        const lo = p.data?.win_rate_ci_lower, hi = p.data?.win_rate_ci_upper
        return lo != null && hi != null ? (lo + hi) / 2 : null
      },
      cellRenderer: (p: { data?: ApiBacktest }) => {
        const lo = p.data?.win_rate_ci_lower, hi = p.data?.win_rate_ci_upper
        if (lo == null || hi == null) return '—'
        return <span style={{ fontSize: '0.78rem', color: '#94a3b8' }}>{(lo * 100).toFixed(1)}% – {(hi * 100).toFixed(1)}%</span>
      },
    },
  ]

  const columnDefs = useMemo((): ColDef<ApiBacktest>[] => {
    if (!scoreConfig || scoreConfig.weights.length === 0) return baseColumnDefs
    return [...baseColumnDefs, {
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
  }, [baseColumnDefs, scoreConfig])

  return (
    <BtkAgGrid
      rowData={backtests}
      columnDefs={columnDefs}
      onRowClicked={e => e.data && navigate(`/backtest/${e.data.id}`)}
      rowStyle={{ cursor: 'pointer' }}
      filterPlaceholder="Filter backtests…"
      initialSortColId="sharpe"
      initialSortDir="desc"
      prefKey={studyId ? `grid.study.${studyId}` : undefined}
    />
  )
}

// ── Heatmap ───────────────────────────────────────────────────────────────────

const LBL = {
  fontSize: '0.68rem', fontWeight: 600, letterSpacing: '0.05em',
  textTransform: 'uppercase' as const, color: '#94a3b8', whiteSpace: 'nowrap' as const,
}
const SEP = { borderLeft: '1px solid #2a3245', height: 20, alignSelf: 'center' as const }

function HeatmapPanel({
  config, onUpdate, onRemove, removable, backtests, sweepAxes, strategies,
}: {
  config:     HeatmapConfig
  onUpdate:   (u: Partial<HeatmapConfig>) => void
  onRemove:   () => void
  removable:  boolean
  backtests:  ApiBacktest[]
  sweepAxes:  string[]
  strategies: string[]
}) {
  const metricDef = HEAT_METRICS.find(m => m.key === config.metric)!
  const constAxes = sweepAxes.filter(p => p !== config.xAxis && p !== config.yAxis)

  const filtered = useMemo(() => {
    const byStrategy = config.strategy
      ? backtests.filter(b => b.strategy_name === config.strategy)
      : backtests
    return byStrategy.filter(b =>
      b.status === 'completed' &&
      constAxes.every(p => {
        const def = getAxisValues(backtests, p)[0] ?? ''
        return valEq(b.params[p], config.constants[p] ?? def)
      })
    )
  }, [backtests, config.strategy, config.constants, constAxes])

  const xVals = useMemo(() => getAxisValues(backtests, config.xAxis), [backtests, config.xAxis])
  const yVals = useMemo(() => getAxisValues(backtests, config.yAxis), [backtests, config.yAxis])

  const z = yVals.map(yv =>
    xVals.map(xv => {
      const b = filtered.find(b => valEq(b.params[config.xAxis], xv) && valEq(b.params[config.yAxis], yv))
      return b ? getMetricValue(b, config.metric) : null
    })
  )
  const zText = z.map(row => row.map(v => v !== null ? metricDef.fmt(v) : ''))

  const heatTrace = {
    type:         'heatmap',
    x:            xVals,
    y:            yVals,
    z,
    colorscale:   metricDef.higherBetter ? SCALE_GOOD : SCALE_BAD,
    text:         zText,
    texttemplate: '%{text}',
    textfont:     { color: '#ffffff', size: 11 },
    hovertemplate: `${config.xAxis}=%{x}<br>${config.yAxis}=%{y}<br>${metricDef.label}: %{text}<extra></extra>`,
    showscale:    true,
    zsmooth:      false,
    colorbar:     { tickfont: { color: '#94a3b8', size: 9 }, outlinecolor: '#2a3245', thickness: 10, len: 0.9 },
  }

  const heatLayout = {
    ...DARK_LAYOUT,
    margin: { t: 8, r: 60, b: 54, l: 70 },
    xaxis: { ...DARK_LAYOUT.xaxis, type: 'category', title: { text: config.xAxis, font: { size: 10 } } },
    yaxis: { ...DARK_LAYOUT.yaxis, type: 'category', title: { text: config.yAxis, font: { size: 10 } } },
  }

  function changeAxis(axis: 'xAxis' | 'yAxis', val: string) {
    const other    = axis === 'xAxis' ? 'yAxis' : 'xAxis'
    const otherVal = val === config[other] ? config[axis] : config[other]
    const remaining = sweepAxes.filter(p => p !== val && p !== otherVal)
    const newConsts = Object.fromEntries(
      remaining.map(p => [p, config.constants[p] ?? (getAxisValues(backtests, p)[0] ?? '')])
    )
    onUpdate({ [axis]: val, [other]: otherVal, constants: newConsts })
  }

  return (
    <div className="btk-chart-card mb-2">
      <div className="d-flex flex-wrap align-items-center gap-2 px-3 py-2" style={{ borderBottom: '1px solid #2a3245' }}>

        {strategies.length > 1 && (
          <>
            <div className="d-flex align-items-center gap-1">
              <span style={LBL}>Strategy</span>
              <select className="form-select btk-metric-select" style={{ minWidth: 0 }}
                value={config.strategy ?? ''} onChange={e => onUpdate({ strategy: e.target.value || null })}>
                <option value="">All</option>
                {strategies.map(s => <option key={s} value={s}>{s.replace(/_/g, ' ')}</option>)}
              </select>
            </div>
            <div style={SEP} />
          </>
        )}

        <div className="d-flex align-items-center gap-1">
          <span style={LBL}>X</span>
          <select className="form-select btk-metric-select" style={{ minWidth: 0 }}
            value={config.xAxis} onChange={e => changeAxis('xAxis', e.target.value)}>
            {sweepAxes.filter(p => p !== config.yAxis).map(p => <option key={p} value={p}>{p}</option>)}
          </select>
        </div>

        <div className="d-flex align-items-center gap-1">
          <span style={LBL}>Y</span>
          <select className="form-select btk-metric-select" style={{ minWidth: 0 }}
            value={config.yAxis} onChange={e => changeAxis('yAxis', e.target.value)}>
            {sweepAxes.filter(p => p !== config.xAxis).map(p => <option key={p} value={p}>{p}</option>)}
          </select>
        </div>

        <div className="d-flex align-items-center gap-1">
          <span style={LBL}>Metric</span>
          <select className="form-select btk-metric-select" style={{ minWidth: 0 }}
            value={config.metric} onChange={e => onUpdate({ metric: e.target.value as HeatMetric })}>
            {HEAT_METRICS.map(m => <option key={m.key} value={m.key}>{m.label}</option>)}
          </select>
        </div>

        {constAxes.length > 0 && <div style={SEP} />}
        {constAxes.map(param => (
          <div key={param} className="d-flex align-items-center gap-1">
            <span style={{ ...LBL, color: '#64748b' }}>fix {param}</span>
            <select className="form-select btk-metric-select" style={{ minWidth: 0 }}
              value={config.constants[param] ?? (getAxisValues(backtests, param)[0] ?? '')}
              onChange={e => onUpdate({ constants: { ...config.constants, [param]: e.target.value } })}>
              {getAxisValues(backtests, param).map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </div>
        ))}

        {removable && (
          <button className="ms-auto btn btn-sm"
            style={{ background: 'transparent', border: '1px solid #2a3245', color: '#64748b', padding: '2px 10px', fontSize: '0.8rem', lineHeight: 1.6 }}
            onClick={onRemove}>×</button>
        )}
      </div>

      <PlotlyChart
        data={[heatTrace]}
        layout={{ ...heatLayout, autosize: true }}
        style={{ width: '100%', height: 260 }}
      />
    </div>
  )
}

function HeatmapSection({ backtests, sweepAxes, strategies }: {
  backtests:  ApiBacktest[]
  sweepAxes:  string[]
  strategies: string[]
}) {
  const nextId = useRef(2)
  const defaultStrategy = strategies.length === 1 ? strategies[0] : null

  function defaultConfig(id: number): HeatmapConfig {
    const x    = sweepAxes[0] ?? ''
    const y    = sweepAxes[1] ?? sweepAxes[0] ?? ''
    const rest = sweepAxes.filter(p => p !== x && p !== y)
    return {
      id,
      xAxis: x,
      yAxis: y,
      metric: 'sharpe',
      constants: Object.fromEntries(rest.map(p => [p, getAxisValues(backtests, p)[0] ?? ''])),
      strategy: defaultStrategy,
    }
  }

  const [panels, setPanels] = useState<HeatmapConfig[]>(() => [defaultConfig(1)])

  function addPanel() {
    const id = nextId.current++
    setPanels(ps => [...ps, defaultConfig(id)])
  }
  function removePanel(id: number) { setPanels(ps => ps.filter(p => p.id !== id)) }
  function updatePanel(id: number, u: Partial<HeatmapConfig>) {
    setPanels(ps => ps.map(p => p.id === id ? { ...p, ...u } : p))
  }

  return (
    <div className="mb-3">
      <div className="d-flex align-items-center justify-content-between mb-2">
        <span className="btk-chart-title">Parameter Heatmaps</span>
        <span className="btk-summary">{panels.length} panel{panels.length !== 1 ? 's' : ''}</span>
      </div>

      {panels.map(panel => (
        <HeatmapPanel
          key={panel.id}
          config={panel}
          onUpdate={u => updatePanel(panel.id, u)}
          onRemove={() => removePanel(panel.id)}
          removable={panels.length > 1}
          backtests={backtests}
          sweepAxes={sweepAxes}
          strategies={strategies}
        />
      ))}

      <button
        onClick={addPanel}
        style={{
          width: '100%', background: 'transparent', cursor: 'pointer',
          border: '1px dashed #2a3245', borderRadius: 10, padding: '9px 0',
          color: '#64748b', fontSize: '0.825rem', transition: 'color 0.15s, border-color 0.15s',
        }}
        onMouseEnter={e => { (e.currentTarget as HTMLElement).style.color = '#94a3b8'; (e.currentTarget as HTMLElement).style.borderColor = '#475569' }}
        onMouseLeave={e => { (e.currentTarget as HTMLElement).style.color = '#64748b'; (e.currentTarget as HTMLElement).style.borderColor = '#2a3245' }}
      >
        <i className="bi bi-plus-lg me-1" />Add Heatmap
      </button>
    </div>
  )
}

// ── Daily P&L Panel ──────────────────────────────────────────────────────────

const PNL_SCALE = [
  [0.00, 'rgba(185,28,28,0.90)'],
  [0.35, 'rgba(220,38,38,0.75)'],
  [0.50, '#1e293b'],
  [0.65, 'rgba(21,128,61,0.75)'],
  [1.00, 'rgba(22,163,74,0.90)'],
]

const WORST_DAY_COLS: ColDef<DailyPnl>[] = [
  { field: 'trade_date',  headerName: 'Date',         width: 110, sort: 'asc' },
  {
    field: 'total_pnl',
    headerName: 'Total P&L',
    width: 130,
    valueFormatter: p => p.value == null ? '—' : `${p.value >= 0 ? '+' : ''}$${p.value.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`,
    cellStyle: p => ({ color: (p.value == null ? '#94a3b8' : p.value >= 0 ? '#4ade80' : '#f87171') as string }),
  },
  { field: 'n_backtests', headerName: '# Backtests',  width: 110 },
  { field: 'n_trades',    headerName: '# Trades',      width: 100 },
]

function DailyPnlPanel({ rows }: { rows: DailyPnl[] }) {
  if (rows.length === 0) return null

  const DAY_LABELS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']

  const dates = rows.map(r => new Date(r.trade_date + 'T12:00:00'))
  const minDate = dates[0]
  // Start from Monday of the first week
  const startDay = minDate.getDay() // 0=Sun, 1=Mon..
  const mondayOffset = startDay === 0 ? -6 : 1 - startDay
  const weekStart = new Date(minDate)
  weekStart.setDate(weekStart.getDate() + mondayOffset)

  const maxWeek = Math.ceil(
    (dates[dates.length - 1].getTime() - weekStart.getTime()) / (7 * 86400000)
  ) + 1

  // z[dayIndex][weekIndex], null = no trade
  const z: (number | null)[][] = Array.from({ length: 5 }, () => Array(maxWeek).fill(null))

  for (let i = 0; i < rows.length; i++) {
    const d = dates[i]
    const dow = d.getDay() // 0=Sun .. 6=Sat
    const dayIdx = dow === 0 ? 6 : dow - 1 // Mon=0..Sun=6; we only use 0-4
    if (dayIdx > 4) continue
    const weekIdx = Math.floor((d.getTime() - weekStart.getTime()) / (7 * 86400000))
    z[dayIdx][weekIdx] = rows[i].total_pnl
  }

  // Month tick labels on x-axis at first week of each month
  const tickvals: number[] = []
  const ticktext: string[] = []
  let lastMonth = -1
  const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
  for (let w = 0; w < maxWeek; w++) {
    const wDate = new Date(weekStart.getTime() + w * 7 * 86400000)
    if (wDate.getMonth() !== lastMonth) {
      lastMonth = wDate.getMonth()
      const yr = wDate.getFullYear().toString().slice(2)
      tickvals.push(w)
      ticktext.push(`${MONTHS[lastMonth]} '${yr}`)
    }
  }

  const allPnl = rows.map(r => Math.abs(r.total_pnl))
  const maxAbs  = Math.max(...allPnl, 1)

  const heatmapTrace = {
    type: 'heatmap',
    x: Array.from({ length: maxWeek }, (_, i) => i),
    y: DAY_LABELS,
    z,
    zmin: -maxAbs,
    zmax:  maxAbs,
    colorscale: PNL_SCALE,
    showscale: true,
    hoverongaps: false,
    xgap: 2,
    ygap: 2,
    colorbar: {
      thickness: 10,
      len: 0.8,
      tickfont: { color: '#94a3b8', size: 10 },
      tickformat: ',.0f',
    },
    hovertemplate: 'Week %{x}<br>%{y}<br>P&L: $%{z:,.0f}<extra></extra>',
  }

  const layout = {
    ...DARK_LAYOUT,
    margin: { t: 8, r: 60, b: 28, l: 46 },
    xaxis: {
      ...DARK_LAYOUT.xaxis,
      tickvals,
      ticktext,
      tickangle: -35,
      showgrid: false,
    },
    yaxis: {
      ...DARK_LAYOUT.yaxis,
      tickfont: { size: 10 },
      showgrid: false,
    },
  }

  const worstRows = [...rows].sort((a, b) => a.total_pnl - b.total_pnl).slice(0, 20)

  return (
    <div className="btk-chart-card mb-3">
      <div className="px-3 pt-3 pb-1">
        <span className="btk-chart-title">Daily P&amp;L — Aggregated Across Study</span>
      </div>
      <PlotlyChart
        data={[heatmapTrace]}
        layout={layout}
        style={{ width: '100%', height: 180 }}
      />
      <div className="px-3 pb-3 pt-2">
        <div style={{ color: '#64748b', fontSize: '0.72rem', marginBottom: 6 }}>Worst 20 days</div>
        <BtkAgGrid<DailyPnl>
          rowData={worstRows}
          columnDefs={WORST_DAY_COLS}
          pageSize={20}
          filterPlaceholder="Filter days…"
        />
      </div>
    </div>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function Study() {
  const { id }              = useParams<{ id: string }>()
  const [study, setStudy]       = useState<ApiStudy | null>(null)
  const [equity, setEquity]     = useState<EquityCurve[]>([])
  const [dailyPnl, setDailyPnl] = useState<DailyPnl[]>([])
  const [loading, setLoading]   = useState(true)
  const [error, setError]       = useState<string | null>(null)
  const [histKey, setHistKey] = useState('sharpe')
  const [xMode, setXMode]     = useState<'date' | 'trade'>('date')
  const [activeTagFilter, setActiveTagFilter] = useState<Set<number>>(new Set())
  const [scoreConfig, setScoreConfig] = useState<ScoreConfig | null>(null)

  useEffect(() => {
    if (!id) return
    Promise.all([
      fetch(`/api/studies/${id}`).then(r => { if (!r.ok) throw new Error(`${r.status}`); return r.json() }),
      fetch(`/api/studies/${id}/equity`).then(r => { if (!r.ok) throw new Error(`${r.status}`); return r.json() }),
      fetch(`/api/studies/${id}/daily-pnl`).then(r => r.ok ? r.json() : []),
    ])
      .then(([s, e, d]) => { setStudy(s); setEquity(e); setDailyPnl(d) })
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false))
  }, [id])

  const updateBacktestTags = useCallback((backtestId: number, tags: Tag[]) => {
    setStudy(prev => {
      if (!prev) return prev
      return {
        ...prev,
        backtests: prev.backtests.map(b => b.id === backtestId ? { ...b, tags } : b),
      }
    })
  }, [])

  // Collect distinct tags across all backtests for the filter bar
  const tagsInView = useMemo((): Tag[] => {
    const seen = new Map<number, Tag>()
    for (const b of study?.backtests ?? []) {
      for (const tag of b.tags ?? []) {
        if (!seen.has(tag.id)) seen.set(tag.id, tag)
      }
    }
    return [...seen.values()].sort((a, b) => a.name.localeCompare(b.name))
  }, [study])

  const filteredBacktests = useMemo(() => {
    const all = study?.backtests ?? []
    if (activeTagFilter.size === 0) return all
    return all.filter(b => (b.tags ?? []).some(t => activeTagFilter.has(t.id)))
  }, [study, activeTagFilter])

  const completed = useMemo(
    () => filteredBacktests.filter(b => b.status === 'completed'),
    [filteredBacktests]
  )

  const bestBt = useMemo(
    () => completed.reduce<ApiBacktest | null>((a, b) =>
      a === null ? b : (b.sharpe ?? -Infinity) > (a.sharpe ?? -Infinity) ? b : a,
      null
    ),
    [completed]
  )

  const filteredIds = useMemo(() => new Set(filteredBacktests.map(b => b.id)), [filteredBacktests])

  // Equity traces from API equity curves
  const equityTraces = useMemo(() => {
    const btById = new Map((study?.backtests ?? []).map(b => [b.id, b]))
    const visibleEquity = activeTagFilter.size > 0
      ? equity.filter(c => filteredIds.has(c.backtest_id))
      : equity
    const traces: unknown[] = visibleEquity.map(c => {
      const initial = c.initial_equity || 100_000
      const y = [initial, ...c.cum_pnl.map(p => initial + p)]
      const isBest = c.backtest_id === bestBt?.id
      let x: (number | string)[]
      if (xMode === 'date' && c.exit_dates.length > 0) {
        const bt = btById.get(c.backtest_id)
        const startDate = bt?.start_date ?? c.exit_dates[0]
        x = [startDate, ...c.exit_dates]
      } else {
        x = Array.from({ length: y.length }, (_, i) => i)
      }
      return {
        x, y,
        type: 'scatter', mode: 'lines',
        line: { width: isBest ? 2.5 : 1, color: isBest ? '#2563EB' : 'rgba(100,116,139,0.25)' },
        hovertemplate: `$%{y:,.0f}<extra>#${c.backtest_id}</extra>`,
      }
    })
    if (traces.length > 0 && xMode !== 'date') {
      const maxLen = Math.max(...visibleEquity.map(c => c.cum_pnl.length + 1))
      const initial = visibleEquity[0]?.initial_equity || 100_000
      traces.unshift({
        x: [0, maxLen - 1], y: [initial, initial],
        type: 'scatter', mode: 'lines',
        line: { width: 1, color: 'rgba(100,116,139,0.2)', dash: 'dot' },
        hoverinfo: 'skip',
      })
    }
    return traces
  }, [equity, bestBt, xMode, study, activeTagFilter, filteredIds])

  // Distribution histogram
  const histMetaDef = HIST_METRICS.find(m => m.key === histKey)!
  const histTrace = useMemo(() => {
    const vals = completed.map(b => getHistValue(b, histKey)).filter((v): v is number => v !== null)
    return {
      x: vals, type: 'histogram',
      marker: { color: 'rgba(37,99,235,0.6)', line: { color: '#2563EB', width: 1 } },
      hovertemplate: '%{x}<br>Count: %{y}<extra></extra>',
      autobinx: true,
    }
  }, [completed, histKey])

  const histLayout = useMemo(() => ({
    ...DARK_LAYOUT,
    xaxis: { ...DARK_LAYOUT.xaxis, title: { text: histMetaDef?.label, font: { size: 10 } }, tickformat: histMetaDef?.tickformat },
    yaxis: { ...DARK_LAYOUT.yaxis, title: { text: 'Count', font: { size: 10 } } },
    bargap: 0.08,
  }), [histKey, histMetaDef])

  const equityLayout = {
    ...DARK_LAYOUT,
    yaxis: { ...DARK_LAYOUT.yaxis, tickformat: '$,.0f', title: { text: 'Equity', font: { size: 10 } } },
    xaxis: xMode === 'date'
      ? { ...DARK_LAYOUT.xaxis, type: 'date', title: { text: 'Date', font: { size: 10 } } }
      : { ...DARK_LAYOUT.xaxis, title: { text: 'Trade #', font: { size: 10 } } },
    hovermode: 'closest',
  }

  const studyStatus: Status = !study ? 'running'
    : study.n_running > 0    ? 'running'
    : study.n_failed > 0 && study.n_completed === 0 ? 'error'
    : 'completed'

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
            <li className="breadcrumb-item active" aria-current="page">
              {loading ? 'Loading…' : (study?.name ?? `Study ${id}`)}
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

        {!loading && !error && study && (
          <>
            {/* Study header */}
            <div className="d-flex align-items-start justify-content-between gap-3 mb-3">
              <div>
                <h5 className="mb-1 fw-semibold" style={{ color: '#e2e8f0' }}>{study.name}</h5>
                <div className="btk-item-sub">
                  <span style={{ color: '#60a5fa', fontFamily: 'monospace', fontSize: '0.75rem' }}>
                    {study.strategy_labels.join(', ') || study.strategies.join(', ')}
                  </span>
                  {study.sweep_axes.length > 0 && (
                    <><span className="mx-2">·</span><span>{study.sweep_axes.join(' × ')}</span></>
                  )}
                  {study.data_start && study.data_end && (
                    <><span className="mx-2">·</span><span>{fmtDate(study.data_start)} – {fmtDate(study.data_end)}</span></>
                  )}
                </div>
                {study.note && <div className="btk-item-sub mt-1" style={{ fontStyle: 'italic' }}>{study.note}</div>}
              </div>
              <StatusBadge status={studyStatus} />
            </div>

            {/* Summary stats */}
            <div className="row g-2 mb-4">
              <div className="col-6 col-sm-4 col-lg-2" style={{ minWidth: 120 }}>
                <div className="btk-metric-card">
                  <div className="btk-metric-label">Backtests</div>
                  <div className="btk-metric-value neutral">{study.n_completed} / {study.total_combinations}</div>
                </div>
              </div>
              {study.best_sharpe != null && (
                <div className="col-6 col-sm-4 col-lg-2" style={{ minWidth: 120 }}>
                  <div className="btk-metric-card">
                    <div className="btk-metric-label">Best Sharpe</div>
                    <div className={`btk-metric-value ${sharpeColor(study.best_sharpe)}`}>{study.best_sharpe.toFixed(2)}</div>
                  </div>
                </div>
              )}
              {study.best_return_pct != null && (
                <div className="col-6 col-sm-4 col-lg-2" style={{ minWidth: 120 }}>
                  <div className="btk-metric-card">
                    <div className="btk-metric-label">Best Return</div>
                    <div className={`btk-metric-value ${study.best_return_pct >= 0 ? 'positive' : 'negative'}`}>
                      {fmt(study.best_return_pct, study.best_return_pct >= 0 ? '+' : '', '%', 1)}
                    </div>
                  </div>
                </div>
              )}
              <div className="col-6 col-sm-4 col-lg-2" style={{ minWidth: 120 }}>
                <div className="btk-metric-card">
                  <div className="btk-metric-label">Total Trades</div>
                  <div className="btk-metric-value neutral">{study.total_trades.toLocaleString()}</div>
                </div>
              </div>
              {study.data_start && study.data_end && (
                <div className="col-6 col-sm-4 col-lg-2" style={{ minWidth: 120 }}>
                  <div className="btk-metric-card">
                    <div className="btk-metric-label">Period</div>
                    <div className="btk-metric-value neutral" style={{ fontSize: '0.85rem' }}>
                      {fmtDate(study.data_start)} – {fmtDate(study.data_end)}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Charts row */}
            <div className="row g-3 mb-3">
              <div className="col-lg-7">
                <div className="btk-chart-card h-100">
                  <div className="d-flex align-items-center justify-content-between px-3 pt-3 pb-1" style={{ flexWrap: 'wrap', gap: 8 }}>
                    <span className="btk-chart-title">Equity Curves</span>
                    <div className="d-flex align-items-center gap-3">
                      <span className="btk-item-sub" style={{ fontSize: '0.72rem' }}>
                        <span style={{ display: 'inline-block', width: 20, height: 2, background: '#2563EB', verticalAlign: 'middle', marginRight: 4 }} />
                        best · <span style={{ color: 'rgba(100,116,139,0.6)' }}>all others</span>
                      </span>
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
                  {equityTraces.length > 1 ? (
                    <PlotlyChart
                      data={equityTraces}
                      layout={{ ...equityLayout, autosize: true }}
                      style={{ width: '100%', height: 290 }}
                    />
                  ) : (
                    <div className="btk-empty" style={{ height: 290 }}>
                      <i className="bi bi-info-circle" />No equity data
                    </div>
                  )}
                </div>
              </div>

              <div className="col-lg-5">
                <div className="btk-chart-card h-100">
                  <div className="d-flex align-items-center justify-content-between px-3 pt-3 pb-1" style={{ gap: 12 }}>
                    <span className="btk-chart-title" style={{ flexShrink: 0 }}>Distribution</span>
                    <select className="form-select btk-metric-select" value={histKey} onChange={e => setHistKey(e.target.value)}>
                      {HIST_METRICS.map(m => <option key={m.key} value={m.key}>{m.label}</option>)}
                    </select>
                  </div>
                  <PlotlyChart
                    data={[histTrace]}
                    layout={{ ...histLayout, autosize: true }}
                    style={{ width: '100%', height: 290 }}
                  />
                </div>
              </div>
            </div>

            {/* Daily P&L */}
            <DailyPnlPanel rows={dailyPnl} />

            {/* Heatmaps */}
            {study.sweep_axes.length >= 2 && (
              <HeatmapSection
                backtests={filteredBacktests}
                sweepAxes={study.sweep_axes}
                strategies={study.strategies}
              />
            )}

            {/* Backtests table */}
            <div className="d-flex align-items-center justify-content-between mb-2 flex-wrap gap-2">
              <span className="btk-chart-title">
                Backtests
                {activeTagFilter.size > 0 && (
                  <span style={{ fontSize: '0.75rem', fontWeight: 400, color: 'var(--btk-muted-dk)', marginLeft: 8 }}>
                    ({filteredBacktests.length} of {study.backtests.length})
                  </span>
                )}
              </span>
              {tagsInView.length > 0 && (
                <TagFilterBar
                  tags={tagsInView}
                  active={activeTagFilter}
                  onChange={setActiveTagFilter}
                />
              )}
            </div>
            <div className="d-flex align-items-center mb-2" style={{ gap: 8 }}>
              <CompositeScore
                prefKey={`composite.study.${id}`}
                onChange={setScoreConfig}
              />
            </div>
            <BacktestTable
              backtests={filteredBacktests}
              onTagsChanged={updateBacktestTags}
              scoreConfig={scoreConfig}
              studyId={id}
            />
          </>
        )}
      </div>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </>
  )
}
