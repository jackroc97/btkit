import { useEffect, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { type ColDef } from 'ag-grid-community'
import {
  createChart,
  createSeriesMarkers,
  CandlestickSeries,
  HistogramSeries,
  BaselineSeries,
  LineSeries,
  LineStyle,
  CrosshairMode,
  type UTCTimestamp,
} from 'lightweight-charts'
import BtkAgGrid from '../components/BtkAgGrid'

// ── API types ─────────────────────────────────────────────────────────────────

interface Leg {
  id: number
  instrument_id: number
  symbol: string
  expiration: string | null
  strike_price: number
  right: string
  action: string
  quantity: number
  multiplier: number
  open_price: number | null
  exit_price: number | null
  entry_delta: number | null
  entry_iv: number | null
  entry_gamma: number | null
  entry_theta: number | null
  entry_vega: number | null
  entry_dte: number | null
}

interface Position {
  id: number
  backtest_id: number
  trade_name: string | null
  open_time: string | null
  exit_time: string | null
  exit_reason: string | null
  duration_min: number | null
  net_pnl: number
  strategy_name: string
  strategy_label: string
  initial_equity: number | null
  credit_received: number
  take_profit_dollars: number | null
  stop_loss_dollars: number | null
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  params: Record<string, any>
  legs: Leg[]
}

interface ChartData {
  has_data: boolean
  candles:      { time: number; open: number; high: number; low: number; close: number }[]
  volume:       { time: number; value: number; color: string }[]
  markers:      { time: number; position: string; color: string; shape: string; text: string }[]
  be_lines:     { price: number; label: string }[]
  strike_lines: { price: number; label: string }[]
  pnl:          { time: number; value: number }[]
  after_exit:   { time: number; value: number }[]
  tp_sl_lines:  { value: number; label: string; color: string }[]
  open_ts: number
  exit_ts: number
  leg_mode: boolean
}

// ── Shared chart theme ────────────────────────────────────────────────────────

const THEME = {
  layout: {
    background: { color: '#161b27' },
    textColor:  '#94a3b8',
    fontSize:   11,
    attributionLogo: false,
  },
  grid: {
    vertLines: { color: '#2a3245' },
    horzLines: { color: '#2a3245' },
  },
  crosshair:       { mode: CrosshairMode.Normal },
  rightPriceScale: { borderColor: '#2a3245' },
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtDateTime(iso: string | null): string {
  if (!iso) return '—'
  const d = new Date(iso)
  return (
    d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }) +
    '  ' +
    d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false })
  )
}

function fmtPnl(v: number): string {
  return `${v >= 0 ? '+' : '-'}$${Math.abs(v).toFixed(0)}`
}

function fmtNum(v: number | null, decimals = 2): string {
  return v == null ? '—' : v.toFixed(decimals)
}

function fmtPct(v: number | null): string {
  return v == null ? '—' : `${(v * 100).toFixed(1)}%`
}

// Forward-fill P&L across the full candle domain so both charts share the same
// logical time axis. Outside the trade window emit whitespace (no value field).
function buildPaddedPnl(
  candles: ChartData['candles'],
  pnlData: ChartData['pnl'],
  openTs: number,
  exitTs: number,
// eslint-disable-next-line @typescript-eslint/no-explicit-any
): any[] {
  const pnlMap = new Map(pnlData.map(p => [p.time, p.value]))
  let last: number | null = null
  return candles.map(c => {
    const known = pnlMap.get(c.time)
    if (known !== undefined) last = known
    if (c.time >= openTs && c.time <= exitTs && last !== null) {
      return { time: c.time as UTCTimestamp, value: last }
    }
    return { time: c.time as UTCTimestamp }   // whitespace — no value field
  })
}

// ── Leg table columns ─────────────────────────────────────────────────────────

const LEG_COLS: ColDef<Leg>[] = [
  {
    field: 'symbol', headerName: 'Symbol', minWidth: 110,
    cellStyle: { fontFamily: 'ui-monospace, monospace', fontSize: '0.8rem', color: '#93c5fd' },
  },
  {
    field: 'right', headerName: 'Right', width: 72,
    cellStyle: p => ({ color: p.value === 'C' ? '#4ade80' : '#f87171', fontWeight: 700, fontFamily: 'monospace' }),
  },
  {
    field: 'action', headerName: 'Action', width: 80,
    cellStyle: p => ({ color: (p.value as string).startsWith('S') ? '#4ade80' : '#f87171', fontFamily: 'monospace', fontWeight: 600 }),
  },
  {
    field: 'strike_price', headerName: 'Strike', width: 80,
    valueFormatter: p => (p.value as number).toFixed(0),
  },
  {
    field: 'expiration', headerName: 'Expiry', minWidth: 90,
    valueFormatter: p => (p.value as string | null) ?? '—',
  },
  { field: 'quantity',   headerName: 'Qty',  width: 64 },
  { field: 'multiplier', headerName: 'Mult', width: 64 },
  { field: 'open_price',  headerName: 'Open $',  width: 80, valueFormatter: p => fmtNum(p.value as number | null) },
  { field: 'exit_price',  headerName: 'Exit $',  width: 80, valueFormatter: p => fmtNum(p.value as number | null) },
  { field: 'entry_delta', headerName: 'δ Delta', width: 84, valueFormatter: p => fmtNum(p.value as number | null, 3) },
  { field: 'entry_iv',    headerName: 'IV',      width: 72, valueFormatter: p => fmtPct(p.value as number | null) },
  { field: 'entry_theta', headerName: 'θ Theta', width: 84, valueFormatter: p => fmtNum(p.value as number | null, 3) },
  { field: 'entry_gamma', headerName: 'γ Gamma', width: 84, valueFormatter: p => fmtNum(p.value as number | null, 4) },
  { field: 'entry_vega',  headerName: 'ν Vega',  width: 80, valueFormatter: p => fmtNum(p.value as number | null, 3) },
  { field: 'entry_dte',   headerName: 'DTE',     width: 64, valueFormatter: p => fmtNum(p.value as number | null, 0) },
]

// ── Navbar ────────────────────────────────────────────────────────────────────

function Navbar() {
  return (
    <nav className="btk-navbar navbar px-3 d-flex align-items-center gap-3">
      <Link to="/" className="btk-brand" style={{ textDecoration: 'none' }}>
        bt<span className="btk-brand-dot">.</span>kit
      </Link>
      <span className="btk-version">v2.0.0</span>
    </nav>
  )
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function Trade() {
  const { id, tradeId } = useParams()
  const [position, setPosition]         = useState<Position | null>(null)
  const [chartData, setChartData]       = useState<ChartData | null>(null)
  const [loading, setLoading]           = useState(true)
  const [chartLoading, setChartLoading] = useState(false)
  const [error, setError]               = useState<string | null>(null)
  const [selectedLeg, setSelectedLeg]   = useState<'spread' | number>('spread')
  const isFirstLegEffect                = useRef(true)

  const priceRef = useRef<HTMLDivElement>(null)
  const pnlRef   = useRef<HTMLDivElement>(null)

  // ── Initial load: position + default chart in parallel ───────────────────
  useEffect(() => {
    if (!tradeId) return
    Promise.all([
      fetch(`/api/positions/${tradeId}`).then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),
      fetch(`/api/positions/${tradeId}/chart`).then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }),
    ])
      .then(([pos, chart]) => {
        setPosition(pos)
        setChartData(chart)
      })
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false))
  }, [tradeId])

  // ── Re-fetch chart when leg selection changes ────────────────────────────
  useEffect(() => {
    if (isFirstLegEffect.current) { isFirstLegEffect.current = false; return }
    if (!tradeId) return
    setChartLoading(true)
    const qs = selectedLeg !== 'spread' ? `?leg_id=${selectedLeg}` : ''
    fetch(`/api/positions/${tradeId}/chart${qs}`)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then(chart => setChartData(chart))
      .catch(e => setError(String(e)))
      .finally(() => setChartLoading(false))
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedLeg])

  // ── LWC chart effect ─────────────────────────────────────────────────────
  useEffect(() => {
    if (!chartData?.has_data || !priceRef.current || !pnlRef.current) return
    const priceEl = priceRef.current
    const pnlEl   = pnlRef.current

    const priceChart = createChart(priceEl, {
      ...THEME,
      width:  priceEl.offsetWidth,
      height: 320,
      timeScale: {
        borderColor:    '#2a3245',
        visible:        false,
        timeVisible:    true,
        secondsVisible: false,
      },
    })

    const pnlChart = createChart(pnlEl, {
      ...THEME,
      width:  pnlEl.offsetWidth,
      height: 180,
      timeScale: {
        borderColor:    '#2a3245',
        timeVisible:    true,
        secondsVisible: false,
      },
    })

    // ── Candlestick + volume ─────────────────────────────────────────────
    const candleSeries = priceChart.addSeries(CandlestickSeries, {
      upColor:         '#4ade80',
      downColor:       '#f87171',
      borderUpColor:   '#4ade80',
      borderDownColor: '#f87171',
      wickUpColor:     '#4ade80',
      wickDownColor:   '#f87171',
    })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    candleSeries.setData(chartData.candles.map(c => ({ ...c, time: c.time as any })))

    const volSeries = priceChart.addSeries(HistogramSeries, {
      priceScaleId: 'vol',
      priceFormat:  { type: 'volume' as const },
    })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    volSeries.setData(chartData.volume.map(v => ({ ...v, time: v.time as any })))
    priceChart.priceScale('vol').applyOptions({ scaleMargins: { top: 0.78, bottom: 0 } })

    // Strike lines on price chart
    for (const sl of chartData.strike_lines) {
      candleSeries.createPriceLine({
        price: sl.price, color: '#475569', lineWidth: 1,
        lineStyle: LineStyle.LargeDashed, axisLabelVisible: true, title: sl.label,
      })
    }
    // Breakeven lines
    for (const be of chartData.be_lines) {
      candleSeries.createPriceLine({
        price: be.price, color: '#6BBCED', lineWidth: 1,
        lineStyle: LineStyle.Dashed, axisLabelVisible: true, title: be.label,
      })
    }

    // Entry / exit markers
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    createSeriesMarkers(candleSeries, chartData.markers.map(m => ({ ...m, time: m.time as any })) as any)

    // ── P&L series — baseline (green above 0, red below 0) ──────────────
    const fmtPnlLabel = (p: number) => `${p >= 0 ? '+' : ''}$${Math.abs(p).toFixed(0)}`
    const pnlSeries = pnlChart.addSeries(BaselineSeries, {
      baseValue:       { type: 'price', price: 0 },
      topLineColor:    '#4ade80',
      topFillColor1:   'rgba(74,222,128,0.12)',
      topFillColor2:   'rgba(74,222,128,0.02)',
      bottomLineColor: '#f87171',
      bottomFillColor1:'rgba(248,113,113,0.02)',
      bottomFillColor2:'rgba(248,113,113,0.12)',
      lineWidth: 2,
      priceFormat: { type: 'custom' as const, minMove: 1, formatter: fmtPnlLabel },
    })

    const paddedPnl = buildPaddedPnl(
      chartData.candles, chartData.pnl, chartData.open_ts, chartData.exit_ts
    )
    pnlSeries.setData(paddedPnl)

    // Dashed gray continuation after exit
    if (chartData.after_exit.length > 1) {
      const contSeries = pnlChart.addSeries(LineSeries, {
        color:     '#475569',
        lineWidth: 2,
        lineStyle: LineStyle.Dashed,
        priceFormat: { type: 'custom' as const, minMove: 1, formatter: fmtPnlLabel },
      })
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      contSeries.setData(chartData.after_exit.map(p => ({ ...p, time: p.time as any })))
    }

    // TP / SL reference lines on P&L chart
    for (const line of chartData.tp_sl_lines) {
      pnlSeries.createPriceLine({
        price: line.value, color: line.color, lineWidth: 1,
        lineStyle: LineStyle.Dashed, axisLabelVisible: true, title: line.label,
      })
    }

    // ── Sync time scales 1:1 ─────────────────────────────────────────────
    let syncing = false
    priceChart.timeScale().subscribeVisibleLogicalRangeChange(range => {
      if (syncing || !range) return
      syncing = true
      pnlChart.timeScale().setVisibleLogicalRange(range)
      syncing = false
    })
    pnlChart.timeScale().subscribeVisibleLogicalRangeChange(range => {
      if (syncing || !range) return
      syncing = true
      priceChart.timeScale().setVisibleLogicalRange(range)
      syncing = false
    })

    // Sync crosshair
    priceChart.subscribeCrosshairMove(param => {
      if (!param.time || !param.point) { pnlChart.clearCrosshairPosition(); return }
      const pnlVal = chartData.pnl.find(p => p.time === param.time)?.value ?? 0
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      pnlChart.setCrosshairPosition(pnlVal, param.time as any, pnlSeries)
    })
    pnlChart.subscribeCrosshairMove(param => {
      if (!param.time || !param.point) priceChart.clearCrosshairPosition()
    })

    // Initial view: open context → a bit past exit
    const { open_ts, exit_ts } = chartData
    const pad = Math.max((exit_ts - open_ts) * 0.08, 900)
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    priceChart.timeScale().setVisibleRange({ from: (open_ts - pad) as any, to: (exit_ts + pad) as any })

    // Responsive resize
    const ro = new ResizeObserver(() => {
      priceChart.applyOptions({ width: priceEl.offsetWidth })
      pnlChart.applyOptions({ width: pnlEl.offsetWidth })
    })
    ro.observe(priceEl)

    return () => {
      ro.disconnect()
      priceChart.remove()
      pnlChart.remove()
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chartData])

  // ── Render ───────────────────────────────────────────────────────────────
  if (loading) {
    return (
      <div data-bs-theme="dark" style={{ minHeight: '100vh', background: '#0f1117' }}>
        <Navbar />
        <div className="container py-5">
          <div className="btk-empty">
            <i className="bi bi-arrow-repeat" style={{ animation: 'spin 1.2s linear infinite', display: 'inline-block' }} />
            Loading…
          </div>
        </div>
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      </div>
    )
  }

  if (error || !position) {
    return (
      <div data-bs-theme="dark" style={{ minHeight: '100vh', background: '#0f1117' }}>
        <Navbar />
        <div className="container py-5">
          <div className="btk-empty" style={{ color: '#f87171' }}>
            <i className="bi bi-exclamation-triangle" />
            {error ?? 'Position not found'}
          </div>
        </div>
      </div>
    )
  }

  const pnlColor   = position.net_pnl >= 0 ? 'positive' : 'negative'
  const exitClass  = (position.exit_reason ?? '').toLowerCase()
  const hasChart   = chartData?.has_data ?? false
  const underlying = position.legs[0]?.symbol?.split(' ')[0] ?? 'ES'

  const selectedLegLabel = selectedLeg !== 'spread'
    ? (() => {
        const leg = position.legs.find(l => l.instrument_id === selectedLeg)
        return leg ? `${leg.action[0]} ${leg.right}${Math.round(leg.strike_price)}` : null
      })()
    : null

  const chartTitle = selectedLegLabel ? `${selectedLegLabel} · 1m` : `${underlying} · 1m`

  return (
    <div data-bs-theme="dark" style={{ minHeight: '100vh', background: '#0f1117' }}>
      <Navbar />

      <div className="container-fluid py-3 px-4" style={{ maxWidth: 1400 }}>

        {/* Breadcrumbs */}
        <nav className="btk-breadcrumb mb-3" aria-label="breadcrumb">
          <ol className="breadcrumb mb-0">
            <li className="breadcrumb-item"><Link to="/">Home</Link></li>
            <li className="breadcrumb-item"><Link to={`/backtest/${id}`}>Backtest #{id}</Link></li>
            <li className="breadcrumb-item active">Trade #{tradeId}</li>
          </ol>
        </nav>

        {/* Stats grid */}
        <div className="row g-2 mb-3">
          <div className="col-6 col-sm-4 col-lg-2" style={{ minWidth: 130 }}>
            <div className="btk-metric-card">
              <div className="btk-metric-label">Strategy</div>
              <div className="btk-metric-value neutral" style={{ fontSize: '0.825rem' }}>{position.strategy_label}</div>
            </div>
          </div>
          <div className="col-6 col-sm-4 col-lg-2" style={{ minWidth: 148 }}>
            <div className="btk-metric-card">
              <div className="btk-metric-label">Entry</div>
              <div className="btk-metric-value neutral" style={{ fontSize: '0.72rem', fontFamily: 'ui-monospace, monospace' }}>
                {fmtDateTime(position.open_time)}
              </div>
            </div>
          </div>
          <div className="col-6 col-sm-4 col-lg-2" style={{ minWidth: 148 }}>
            <div className="btk-metric-card">
              <div className="btk-metric-label">Exit</div>
              <div className="btk-metric-value neutral" style={{ fontSize: '0.72rem', fontFamily: 'ui-monospace, monospace' }}>
                {fmtDateTime(position.exit_time)}
              </div>
            </div>
          </div>
          <div className="col-6 col-sm-4 col-lg-2" style={{ minWidth: 100 }}>
            <div className="btk-metric-card">
              <div className="btk-metric-label">Duration</div>
              <div className="btk-metric-value neutral">
                {position.duration_min != null ? `${position.duration_min} min` : '—'}
              </div>
            </div>
          </div>
          <div className="col-6 col-sm-4 col-lg-2" style={{ minWidth: 100 }}>
            <div className="btk-metric-card">
              <div className="btk-metric-label">Exit Reason</div>
              <div style={{ marginTop: 4 }}>
                <span className={`btk-exit-badge ${exitClass}`}>{position.exit_reason ?? '—'}</span>
              </div>
            </div>
          </div>
          <div className="col-6 col-sm-4 col-lg-2" style={{ minWidth: 100 }}>
            <div className="btk-metric-card">
              <div className="btk-metric-label">P&amp;L</div>
              <div className={`btk-metric-value ${pnlColor}`}>{fmtPnl(position.net_pnl)}</div>
            </div>
          </div>
          {position.credit_received !== 0 && (
            <div className="col-6 col-sm-4 col-lg-2" style={{ minWidth: 100 }}>
              <div className="btk-metric-card">
                <div className="btk-metric-label">Credit</div>
                <div className="btk-metric-value neutral">{fmtPnl(position.credit_received)}</div>
              </div>
            </div>
          )}
          {position.take_profit_dollars != null && (
            <div className="col-6 col-sm-4 col-lg-2" style={{ minWidth: 100 }}>
              <div className="btk-metric-card">
                <div className="btk-metric-label">TP Target</div>
                <div className="btk-metric-value positive">{fmtPnl(position.take_profit_dollars)}</div>
              </div>
            </div>
          )}
          {position.stop_loss_dollars != null && (
            <div className="col-6 col-sm-4 col-lg-2" style={{ minWidth: 100 }}>
              <div className="btk-metric-card">
                <div className="btk-metric-label">SL Limit</div>
                <div className="btk-metric-value negative">{fmtPnl(position.stop_loss_dollars)}</div>
              </div>
            </div>
          )}
        </div>

        {/* Leg selector — sits above the chart card */}
        {position.legs.length > 0 && (
          <div className="d-flex align-items-center mb-2" style={{ gap: 8 }}>
            <div className="d-flex" style={{ background: '#1e2535', borderRadius: 6, padding: 2, gap: 2 }}>
              <button
                onClick={() => setSelectedLeg('spread')}
                style={{
                  background: selectedLeg === 'spread' ? '#2563eb' : 'transparent',
                  border: 'none', borderRadius: 4, padding: '4px 10px',
                  color: selectedLeg === 'spread' ? '#fff' : '#64748b',
                  cursor: 'pointer', fontSize: '0.75rem', fontWeight: 600,
                  transition: 'background 0.15s, color 0.15s',
                }}
              >
                Spread
              </button>
              {position.legs.map(leg => (
                <button
                  key={leg.instrument_id}
                  onClick={() => setSelectedLeg(leg.instrument_id)}
                  style={{
                    background: selectedLeg === leg.instrument_id ? '#2563eb' : 'transparent',
                    border: 'none', borderRadius: 4, padding: '4px 10px',
                    color: selectedLeg === leg.instrument_id ? '#fff' : '#64748b',
                    cursor: 'pointer', fontSize: '0.75rem', fontFamily: 'ui-monospace, monospace',
                    transition: 'background 0.15s, color 0.15s',
                  }}
                >
                  {leg.action[0]} {leg.right}{Math.round(leg.strike_price)}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Chart area */}
        {!hasChart ? (
          <div
            className="btk-chart-card mb-4 d-flex flex-column align-items-center justify-content-center"
            style={{ padding: '48px 24px', gap: 8, textAlign: 'center' }}
          >
            <i className="bi bi-bar-chart-steps" style={{ fontSize: '2rem', color: 'var(--btk-muted-dk)', opacity: 0.4 }} />
            <div style={{ color: 'var(--btk-muted-dk)', fontSize: '0.875rem', fontWeight: 500 }}>
              Intraday price data not available
            </div>
            <div style={{ color: 'var(--btk-muted-dk)', fontSize: '0.775rem' }}>
              Start the server with <code style={{ color: '#93c5fd' }}>BTKIT_INPUT_DB=…</code> to enable candlestick charts.
            </div>
          </div>
        ) : (
          <div className="btk-chart-card mb-4" style={{ position: 'relative' }}>
            {/* Chart loading overlay */}
            {chartLoading && (
              <div style={{
                position: 'absolute', inset: 0, zIndex: 10, borderRadius: 10,
                background: 'rgba(22,27,39,0.55)', display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <i className="bi bi-arrow-repeat" style={{ animation: 'spin 1.2s linear infinite', display: 'inline-block', fontSize: '1.5rem', color: '#64748b' }} />
              </div>
            )}

            {/* Price pane header */}
            <div className="d-flex align-items-center justify-content-between px-3 pt-3 pb-2">
              <span className="btk-chart-title">{chartTitle}</span>
              <div className="d-flex gap-3 align-items-center" style={{ fontSize: '0.72rem', color: '#64748b' }}>
                <span><span style={{ color: '#4ade80' }}>▲</span> Entry</span>
                <span><span style={{ color: chartData?.markers[1]?.color ?? '#94a3b8' }}>▼</span> Exit</span>
              </div>
            </div>
            <div ref={priceRef} />

            {/* P&L pane */}
            <div style={{ borderTop: '1px solid #2a3245' }}>
              <div
                className="d-flex align-items-center gap-3 px-3 pt-2 pb-1"
                style={{ fontSize: '0.68rem', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: '#64748b' }}
              >
                <span>Running P&amp;L{selectedLegLabel ? ` · ${selectedLegLabel}` : ''}</span>
                {(chartData?.tp_sl_lines ?? []).map(l => (
                  <span key={l.label} style={{ color: l.color, fontWeight: 500 }}>
                    — {l.label} {l.value >= 0 ? '+' : ''}${Math.abs(l.value).toFixed(0)}
                  </span>
                ))}
                {(chartData?.after_exit ?? []).length > 1 && (
                  <span style={{ display: 'flex', alignItems: 'center', gap: 4, color: '#475569', fontWeight: 500 }}>
                    <svg width="18" height="2" style={{ marginBottom: 1 }}>
                      <line x1="0" y1="1" x2="18" y2="1" stroke="#475569" strokeWidth="1.5" strokeDasharray="4,3" />
                    </svg>
                    post-exit
                  </span>
                )}
              </div>
              <div ref={pnlRef} />
            </div>
          </div>
        )}

        {/* Legs table */}
        <div className="mb-3">
          <div className="mb-2">
            <span className="btk-chart-title">Legs</span>
          </div>
          {position.legs.length === 0 ? (
            <div className="btk-empty">
              <i className="bi bi-inbox" />
              No leg data found.
            </div>
          ) : (
            <BtkAgGrid
              rowData={position.legs}
              columnDefs={LEG_COLS}
              pageSize={20}
              filterPlaceholder="Filter legs…"
            />
          )}
        </div>

      </div>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  )
}
