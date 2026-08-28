import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import {
  createChart,
  AreaSeries,
  CandlestickSeries,
  HistogramSeries,
  LineSeries,
  CrosshairMode,
} from 'lightweight-charts'

// ── API types ─────────────────────────────────────────────────────────────────

interface BacktestMeta {
  id: number
  strategy_label: string
  initial_equity: number | null
  start_date: string | null
  end_date: string | null
}

interface EquityPosition {
  exit_date: string
  cum_pnl: number
}

interface IndicatorMeta {
  id: number | null
  name: string
  placement: 'overlay' | 'panel'
}

interface ActiveIndicator {
  name: string
  placement: 'overlay' | 'panel'
  color: string
  data: { time: number; value: number }[]
}

interface Candle { time: number; open: number; high: number; low: number; close: number }
interface Bars {
  has_data: boolean
  candles: Candle[]
  volume: { time: number; value: number; color: string }[]
}

// ── Constants ─────────────────────────────────────────────────────────────────

const IND_COLORS = [
  '#60a5fa', '#f59e0b', '#a78bfa', '#34d399',
  '#fb923c', '#e879f9', '#2dd4bf', '#f472b6',
]

const TIMEFRAMES = ['1H', '1D', '1W'] as const
type Timeframe = typeof TIMEFRAMES[number]

// Fixed right-price-scale width, applied to every pane (equity/price/indicator
// panel) so none of them shift horizontally relative to the others. Each pane
// shows very differently-shaped numbers on its own right scale (equity in
// dollars, price in index points, indicators in arbitrary units), so their
// natural label widths differ; a shared floor keeps them pixel-aligned. See
// lightweight-charts' own note on PriceScaleOptions.minimumWidth: "ensuring
// that multiple charts positioned in a vertical stack each have an identical
// price scale width." Set generously so typical labels never exceed it (a
// value that gets exceeded reintroduces the misalignment on just that pane).
const RIGHT_SCALE_WIDTH = 84

const THEME = {
  layout: { background: { color: '#0f1117' }, textColor: '#94a3b8', fontSize: 11, attributionLogo: false },
  grid: { vertLines: { color: '#222a3a' }, horzLines: { color: '#222a3a' } },
  crosshair: { mode: CrosshairMode.Normal },
  rightPriceScale: { borderColor: '#2a3245', minimumWidth: RIGHT_SCALE_WIDTH },
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function dateToUnix(dateStr: string): number {
  return Math.floor(new Date(`${dateStr}T00:00:00Z`).getTime() / 1000)
}

function rangeQs(start: string | null, end: string | null): string {
  return `${start ? `&start=${start}` : ''}${end ? `&end=${end}` : ''}`
}

function indicatorUrl(root: string, name: string, timeframe: string, start: string | null, end: string | null): string {
  return `/api/explore/indicator?timeframe=${timeframe}&name=${encodeURIComponent(name)}&root=${encodeURIComponent(root)}${rangeQs(start, end)}`
}

// Re-sample the sparse per-trade equity checkpoints onto the price chart's own
// candle timestamps (one point per candle, forward-filling the last known
// equity value) — the same technique Trade.tsx's buildPaddedPnl uses to align
// its P&L series to the candle domain. Without this, equity (one point per
// trade) and price (one point per bar) have wildly different bar counts, and
// syncing panes by logical/bar-index range shows a different date range on
// each pane — worse, it can make lightweight-charts clamp the requested range
// hard enough to produce runaway drift/zoom as the sync loop bounces between
// panes.
function alignEquityToCandles(
  candles: { time: number }[],
  checkpoints: { time: number; value: number }[],
  initialEquity: number,
): { time: number; value: number }[] {
  let i = 0
  let current = initialEquity
  return candles.map(c => {
    while (i < checkpoints.length && checkpoints[i].time <= c.time) {
      current = checkpoints[i].value
      i++
    }
    return { time: c.time, value: current }
  })
}

// Re-sample an indicator's {time, value} series onto the price candles' own
// timestamps, same idea as alignEquityToCandles above but leaving candles the
// indicator has no data for as *whitespace* (a bare {time}, matching Trade.tsx
// buildPaddedPnl's convention) rather than forward-filling — an indicator's
// warm-up window (e.g. a rolling z-score needing N prior bars) has no
// meaningful "last known value" to carry forward, so the line should simply
// not exist yet there instead of showing a misleading flat/stale value.
// Without this, a panel indicator has fewer bars than price/equity (the
// backend drops null buckets), so the shared logical-index sync points it at
// the wrong dates as soon as a third pane is added.
function alignIndicatorToCandles(
  candles: { time: number }[],
  points: { time: number; value: number }[],
): ({ time: number; value: number } | { time: number })[] {
  const byTime = new Map(points.map(p => [p.time, p.value]))
  return candles.map(c => {
    const v = byTime.get(c.time)
    return v === undefined ? { time: c.time } : { time: c.time, value: v }
  })
}

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

export default function Equity() {
  const { id } = useParams()
  const [bt, setBt]                   = useState<BacktestMeta | null>(null)
  const [positions, setPositions]     = useState<EquityPosition[]>([])
  const [loading, setLoading]         = useState(true)
  const [error, setError]             = useState<string | null>(null)

  const [showPrice, setShowPrice]     = useState(true)
  const [root, setRoot]               = useState<string | null | undefined>(undefined)
  const [timeframe, setTimeframe]     = useState<Timeframe>('1D')
  const [priceLoading, setPriceLoading] = useState(false)
  const [priceData, setPriceData]     = useState<Bars | null>(null)
  const [availableIndicators, setAvailableIndicators] = useState<IndicatorMeta[]>([])
  const [activeIndicators, setActiveIndicators]       = useState<ActiveIndicator[]>([])

  const equityRef         = useRef<HTMLDivElement>(null)
  const priceRef          = useRef<HTMLDivElement>(null)
  const indicatorPanelRef = useRef<HTMLDivElement>(null)

  // ── Initial load: backtest + positions + underlying root, in parallel ────
  useEffect(() => {
    if (!id) return
    setLoading(true)
    Promise.all([
      fetch(`/api/backtests/${id}`).then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() }),
      fetch(`/api/backtests/${id}/positions`).then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() }),
      fetch(`/api/backtests/${id}/underlying`).then(r => r.json()).catch(() => ({ root: null })),
    ])
      .then(([b, p, u]: [BacktestMeta, EquityPosition[], { root: string | null }]) => {
        setBt(b)
        setPositions(p)
        setRoot(u.root ?? null)
      })
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false))
  }, [id])

  // ── Load bars whenever root/timeframe/range changes ──────────────────────
  useEffect(() => {
    if (!root) return
    setPriceLoading(true)
    fetch(`/api/explore/bars?timeframe=${timeframe}&root=${encodeURIComponent(root)}${rangeQs(bt?.start_date ?? null, bt?.end_date ?? null)}`)
      .then(r => r.json())
      .then((d: Bars) => setPriceData(d))
      .catch(() => setPriceData({ has_data: false, candles: [], volume: [] }))
      .finally(() => setPriceLoading(false))
  }, [root, timeframe, bt?.start_date, bt?.end_date])

  // ── Load available indicators once the root resolves ─────────────────────
  useEffect(() => {
    if (!root) return
    fetch(`/api/explore/indicators?root=${encodeURIComponent(root)}`)
      .then(r => r.json())
      .then((d: { indicators: IndicatorMeta[] }) => setAvailableIndicators(d.indicators ?? []))
      .catch(() => setAvailableIndicators([]))
  }, [root])

  // ── Refetch active indicator data when timeframe changes (data is bucketed) ─
  useEffect(() => {
    if (!root || activeIndicators.length === 0) return
    Promise.all(
      activeIndicators.map(ind =>
        fetch(indicatorUrl(root, ind.name, timeframe, bt?.start_date ?? null, bt?.end_date ?? null))
          .then(r => r.json())
          .then((d: { data: { time: number; value: number }[] }) => ({ ...ind, data: d.data ?? [] }))
      )
    ).then(setActiveIndicators)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [timeframe])

  function handleAddIndicator(meta: IndicatorMeta) {
    if (!root || activeIndicators.find(a => a.name === meta.name)) return
    const color = IND_COLORS[activeIndicators.length % IND_COLORS.length]
    fetch(indicatorUrl(root, meta.name, timeframe, bt?.start_date ?? null, bt?.end_date ?? null))
      .then(r => r.json())
      .then((d: { placement: 'overlay' | 'panel'; data: { time: number; value: number }[] }) => {
        setActiveIndicators(prev => [
          ...prev,
          { name: meta.name, placement: d.placement ?? meta.placement, color, data: d.data ?? [] },
        ])
      })
  }
  function handleRemoveIndicator(name: string) {
    setActiveIndicators(prev => prev.filter(a => a.name !== name))
  }
  function handleSetPlacement(name: string, placement: 'overlay' | 'panel') {
    setActiveIndicators(prev => prev.map(a => a.name === name ? { ...a, placement } : a))
  }

  // ── Equity checkpoints — client-side, one point per trade exit ────────────
  const equityCheckpoints = useMemo(() => {
    const initialEquity = bt?.initial_equity ?? 100_000
    const byTime = new Map<number, number>()
    if (bt?.start_date) byTime.set(dateToUnix(bt.start_date), initialEquity)
    for (const p of positions) {
      if (!p.exit_date) continue
      byTime.set(dateToUnix(p.exit_date), initialEquity + p.cum_pnl)
    }
    return Array.from(byTime.entries())
      .sort((a, b) => a[0] - b[0])
      .map(([time, value]) => ({ time, value }))
  }, [positions, bt])

  // Re-sampled onto the price candles' own timestamps when price is shown, so
  // the two panes share the same bar count/order for logical-range sync — see
  // alignEquityToCandles above.
  const equityAligned = useMemo(() => {
    if (!showPrice || !priceData?.has_data || priceData.candles.length === 0) return null
    return alignEquityToCandles(priceData.candles, equityCheckpoints, bt?.initial_equity ?? 100_000)
  }, [showPrice, priceData, equityCheckpoints, bt?.initial_equity])

  // ── LWC chart lifecycle ───────────────────────────────────────────────────
  useEffect(() => {
    if (equityCheckpoints.length === 0 || !equityRef.current) return
    const equityChartData = equityAligned ?? equityCheckpoints
    const equityEl = equityRef.current
    const priceEl  = showPrice && priceData?.has_data ? priceRef.current : null
    const overlays = activeIndicators.filter(i => i.placement === 'overlay')
    const panels   = activeIndicators.filter(i => i.placement === 'panel')
    const panelEl  = panels.length > 0 ? indicatorPanelRef.current : null

    const fmtTime = (time: number) =>
      new Date(time * 1000).toLocaleDateString('en-US', {
        timeZone: 'America/New_York', month: 'short', day: 'numeric', year: '2-digit',
      })

    const equityChart = createChart(equityEl, {
      ...THEME,
      width: equityEl.clientWidth, height: 320,
      localization: { timeFormatter: fmtTime },
      timeScale: { borderColor: '#2a3245', visible: !priceEl && !panelEl },
    })
    const equitySeriesObj = equityChart.addSeries(AreaSeries, {
      lineColor: '#2563eb', lineWidth: 2,
      topColor: 'rgba(37,99,235,0.16)', bottomColor: 'rgba(37,99,235,0.0)',
      priceFormat: {
        type: 'custom' as const, minMove: 1,
        formatter: (v: number) => `$${v.toLocaleString('en-US', { maximumFractionDigits: 0 })}`,
      },
    })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    equitySeriesObj.setData(equityChartData.map(p => ({ ...p, time: p.time as any })))

    let priceChart: ReturnType<typeof createChart> | null = null
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let candleSeries: any = null
    if (priceEl && priceData) {
      priceChart = createChart(priceEl, {
        ...THEME,
        width: priceEl.clientWidth, height: 260,
        localization: { timeFormatter: fmtTime },
        timeScale: { borderColor: '#2a3245', visible: !panelEl },
      })
      const candle = priceChart.addSeries(CandlestickSeries, {
        upColor: '#4ade80', downColor: '#f87171',
        borderUpColor: '#4ade80', borderDownColor: '#f87171',
        wickUpColor: '#4ade80', wickDownColor: '#f87171',
      })
      candleSeries = candle
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      candle.setData(priceData.candles.map(c => ({ ...c, time: c.time as any })))

      const vol = priceChart.addSeries(HistogramSeries, { priceScaleId: 'vol', priceFormat: { type: 'volume' as const } })
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      vol.setData(priceData.volume.map(v => ({ ...v, time: v.time as any })))
      priceChart.priceScale('vol').applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } })

      // "Overlay" indicators (per the backend's placement heuristic) are in the
      // same units as the underlying price — e.g. a moving average — so they
      // share the candles' own price scale rather than getting a second,
      // independently auto-scaled axis (which wouldn't visually track price
      // and adds an extra axis for no reason).
      for (const ind of overlays) {
        const s = priceChart.addSeries(LineSeries, {
          color: ind.color, lineWidth: 1,
          priceLineVisible: false, lastValueVisible: true,
        })
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        s.setData(ind.data.map(p => ({ ...p, time: p.time as any })))
      }
    }

    let indicatorChart: ReturnType<typeof createChart> | null = null
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let panelAnchorSeries: any = null
    let panelAnchorData: ({ time: number; value: number } | { time: number })[] = []
    if (panelEl && panels.length > 0) {
      indicatorChart = createChart(panelEl, {
        ...THEME,
        width: panelEl.clientWidth, height: 140,
        localization: { timeFormatter: fmtTime },
        timeScale: { borderColor: '#2a3245' },
      })
      for (const ind of panels) {
        const s = indicatorChart.addSeries(LineSeries, {
          color: ind.color, lineWidth: 1, priceLineVisible: false, lastValueVisible: true,
        })
        // Re-sample onto the price candles' timestamps (see
        // alignIndicatorToCandles) so this pane shares the same bar count/order
        // as price/equity — falls back to the raw series if candles aren't
        // loaded yet (a brief window right after enabling an indicator).
        const data = priceData?.has_data
          ? alignIndicatorToCandles(priceData.candles, ind.data)
          : ind.data
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        s.setData(data.map(p => ({ ...p, time: p.time as any })))
        if (!panelAnchorSeries) { panelAnchorSeries = s; panelAnchorData = data }
      }
    }

    // Unified logical (bar-index) time-scale sync across all active charts —
    // same pattern as Trade.tsx. This only lines up correctly because equity is
    // re-sampled onto the price candles' own timestamps above (equityAligned),
    // so every pane here has the same bar count/order; syncing by bar index
    // without that alignment is what caused the earlier drift/zoom feedback.
    const charts = [equityChart, ...(priceChart ? [priceChart] : []), ...(indicatorChart ? [indicatorChart] : [])]
    let syncing = false
    for (const src of charts) {
      src.timeScale().subscribeVisibleLogicalRangeChange(range => {
        if (syncing || !range) return
        syncing = true
        for (const tgt of charts) {
          if (tgt !== src) tgt.timeScale().setVisibleLogicalRange(range)
        }
        syncing = false
      })
    }

    // Crosshair sync: draw a vertical line at the same time across every
    // mounted pane, so hovering any one of them shows which date corresponds
    // to a point on the others. Bidirectional across all panes — unlike
    // Trade.tsx's price→P&L version, which only ever clears (never sets) the
    // price chart's crosshair when hovering the P&L pane below it. Equity
    // shares the exact same timestamps as the price candles when price is
    // shown (equityChartData is re-sampled onto them above), so the lookup
    // always resolves between those two; the indicator panel may be missing a
    // few points at the start (warm-up), in which case its crosshair is
    // simply cleared for that instant instead of showing a stale value.
    type CrosshairPane = {
      chart: ReturnType<typeof createChart>
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      series: any
      byTime: Map<number, number>
    }
    const crosshairPanes: CrosshairPane[] = [
      { chart: equityChart, series: equitySeriesObj, byTime: new Map(equityChartData.map(p => [p.time, p.value])) },
    ]
    if (priceChart && candleSeries) {
      crosshairPanes.push({
        chart: priceChart,
        series: candleSeries,
        byTime: new Map(priceData!.candles.map(c => [c.time, c.close])),
      })
    }
    if (indicatorChart && panelAnchorSeries) {
      crosshairPanes.push({
        chart: indicatorChart,
        series: panelAnchorSeries,
        // Whitespace entries (no `value` — see alignIndicatorToCandles) are
        // dropped here, same as a missing key: the crosshair sync below
        // already clears the target pane's crosshair when a time has no entry.
        byTime: new Map(
          panelAnchorData
            .filter((p): p is { time: number; value: number } => 'value' in p)
            .map(p => [p.time, p.value])
        ),
      })
    }
    for (const src of crosshairPanes) {
      src.chart.subscribeCrosshairMove(param => {
        if (!param.time || !param.point) {
          for (const tgt of crosshairPanes) if (tgt !== src) tgt.chart.clearCrosshairPosition()
          return
        }
        for (const tgt of crosshairPanes) {
          if (tgt === src) continue
          const val = tgt.byTime.get(param.time as unknown as number)
          if (val === undefined) { tgt.chart.clearCrosshairPosition(); continue }
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          tgt.chart.setCrosshairPosition(val, param.time as any, tgt.series)
        }
      })
    }

    // Fit content on whichever pane has the broadest/most granular time domain —
    // the others pick up the same visible range via the sync subscription above.
    // Fitting each chart independently would leave them showing different spans.
    ;(priceChart ?? indicatorChart ?? equityChart).timeScale().fitContent()

    const ro = new ResizeObserver(() => {
      equityChart.applyOptions({ width: equityEl.clientWidth })
      if (priceChart && priceEl) priceChart.applyOptions({ width: priceEl.clientWidth })
      if (indicatorChart && panelEl) indicatorChart.applyOptions({ width: panelEl.clientWidth })
    })
    ro.observe(equityEl)

    return () => {
      ro.disconnect()
      equityChart.remove()
      priceChart?.remove()
      indicatorChart?.remove()
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [equityCheckpoints, equityAligned, showPrice, priceData, activeIndicators])

  const panelInds = activeIndicators.filter(i => i.placement === 'panel')

  return (
    <div data-bs-theme="dark" style={{ minHeight: '100vh', background: '#0f1117' }}>
      <Navbar />

      <div className="container-fluid py-3 px-4" style={{ maxWidth: 1400 }}>

        {/* Breadcrumbs */}
        <nav className="btk-breadcrumb mb-3" aria-label="breadcrumb">
          <ol className="breadcrumb mb-0">
            <li className="breadcrumb-item"><Link to="/">Home</Link></li>
            <li className="breadcrumb-item"><Link to={`/backtest/${id}`}>Backtest #{id}</Link></li>
            <li className="breadcrumb-item active">Equity Curve</li>
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
            {/* Controls */}
            <div className="d-flex align-items-center flex-wrap mb-2" style={{ gap: 8 }}>
              <span style={{ fontSize: '0.9rem', fontWeight: 600, color: '#e2e8f0', marginRight: 6 }}>
                {bt.strategy_label}
              </span>

              <button
                onClick={() => setShowPrice(v => !v)}
                disabled={root === null}
                title={root === null ? 'Underlying price not available — start the server with BTKIT_INPUT_DB=… to enable it.' : 'Toggle underlying price'}
                style={{
                  background: showPrice ? 'rgba(37,99,235,0.2)' : 'transparent',
                  border: `1px solid ${showPrice ? '#2563eb' : '#2a3245'}`,
                  borderRadius: 5, color: root === null ? '#3f4a61' : (showPrice ? '#93c5fd' : '#64748b'),
                  cursor: root === null ? 'not-allowed' : 'pointer', fontSize: '0.72rem', fontWeight: 600,
                  padding: '3px 10px', transition: 'all 0.15s', letterSpacing: '0.02em',
                }}
              >
                <i className="bi bi-graph-up me-1" style={{ fontSize: '0.75rem' }} />
                Underlying Price
              </button>

              {showPrice && root && (
                <div className="d-flex" style={{ background: '#1e2535', borderRadius: 6, padding: 2, gap: 2 }}>
                  {TIMEFRAMES.map(tf => (
                    <button
                      key={tf}
                      onClick={() => setTimeframe(tf)}
                      style={{
                        background: timeframe === tf ? '#2563eb' : 'transparent',
                        border: 'none', borderRadius: 4, padding: '3px 9px',
                        color: timeframe === tf ? '#fff' : '#64748b',
                        cursor: 'pointer', fontSize: '0.72rem', fontWeight: 600, fontFamily: 'ui-monospace, monospace',
                      }}
                    >
                      {tf}
                    </button>
                  ))}
                </div>
              )}

              {showPrice && root && availableIndicators.length > 0 && (
                <select
                  aria-label="Add indicator"
                  value=""
                  onChange={e => {
                    const meta = availableIndicators.find(a => a.name === e.target.value)
                    if (meta) handleAddIndicator(meta)
                    e.currentTarget.value = ''
                  }}
                  style={{
                    background: '#1e2535', border: '1px solid #2a3245', borderRadius: 6,
                    color: '#94a3b8', fontSize: '0.75rem', height: 28, padding: '0 8px', cursor: 'pointer',
                  }}
                >
                  <option value="">+ Indicator…</option>
                  {availableIndicators.filter(a => !activeIndicators.find(i => i.name === a.name)).map(a => (
                    <option key={a.name} value={a.name}>{a.name}</option>
                  ))}
                </select>
              )}

              {activeIndicators.map(ind => (
                <div
                  key={ind.name}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 5,
                    background: '#1e2535', borderRadius: 6, padding: '3px 6px 3px 8px',
                    border: '1px solid #2a3245',
                  }}
                >
                  <span style={{ width: 7, height: 7, borderRadius: '50%', background: ind.color, flexShrink: 0 }} />
                  <span style={{ fontSize: '0.72rem', color: '#94a3b8', fontFamily: 'ui-monospace, monospace' }}>{ind.name}</span>
                  <div style={{ display: 'flex', background: '#0f1117', borderRadius: 4, padding: 1, gap: 1, marginLeft: 2 }}>
                    {(['overlay', 'panel'] as const).map(p => (
                      <button
                        key={p}
                        onClick={() => handleSetPlacement(ind.name, p)}
                        title={p === 'overlay' ? 'Overlay on price chart' : 'Show as sub-chart'}
                        style={{
                          background: ind.placement === p ? '#2563eb' : 'transparent',
                          border: 'none', borderRadius: 3, padding: '2px 5px',
                          color: ind.placement === p ? '#fff' : '#475569',
                          cursor: 'pointer', fontSize: '0.65rem', fontWeight: 600, lineHeight: 1.4,
                        }}
                      >
                        {p === 'overlay' ? '⬆' : '⬇'}
                      </button>
                    ))}
                  </div>
                  <button
                    onClick={() => handleRemoveIndicator(ind.name)}
                    title="Remove indicator"
                    style={{ background: 'none', border: 'none', color: '#475569', cursor: 'pointer', padding: '0 2px', fontSize: '0.85rem', lineHeight: 1 }}
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>

            {/* Chart card */}
            {positions.length === 0 ? (
              <div className="btk-empty" style={{ height: 280 }}>
                <i className="bi bi-info-circle" />No positions
              </div>
            ) : (
              <div className="btk-chart-card mb-4">
                <div ref={equityRef} />

                {showPrice && (
                  <div style={{ borderTop: '1px solid #2a3245' }}>
                    {priceLoading && !priceData ? (
                      <div className="btk-empty" style={{ height: 120 }}>
                        <i className="bi bi-arrow-repeat" style={{ animation: 'spin 1.2s linear infinite', display: 'inline-block' }} />
                        Loading…
                      </div>
                    ) : priceData?.has_data ? (
                      <div ref={priceRef} />
                    ) : (
                      <div className="btk-empty" style={{ height: 120 }}>
                        <i className="bi bi-info-circle" />No underlying price data for this range.
                      </div>
                    )}
                  </div>
                )}

                {panelInds.length > 0 && (
                  <div style={{ borderTop: '1px solid #2a3245' }}>
                    <div
                      className="d-flex align-items-center gap-3 px-3 pt-2 pb-1"
                      style={{ fontSize: '0.68rem', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: '#64748b' }}
                    >
                      {panelInds.map(ind => <span key={ind.name} style={{ color: ind.color }}>— {ind.name}</span>)}
                    </div>
                    <div ref={indicatorPanelRef} />
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  )
}
