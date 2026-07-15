import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  createChart,
  createSeriesMarkers,
  CandlestickSeries,
  HistogramSeries,
  LineSeries,
  BaselineSeries,
  CrosshairMode,
} from 'lightweight-charts'

// ── API types ─────────────────────────────────────────────────────────────────

interface Contract {
  kind: 'continuous' | 'contract'
  symbol: string
  instrument_id?: number
  root?: string
  start: string
  end: string
  bars: number
}

interface IndicatorMeta {
  id: number | null
  name: string
  placement: 'overlay' | 'panel'
}

interface ActiveIndicator {
  id: number | null
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

interface BacktestMeta { id: number; strategy_label: string; strategy_name: string }
interface Overlay {
  markers: { time: number; position: string; color: string; shape: string; text: string }[]
  equity: { time: number; value: number }[]
  n_positions: number
}

// ── Constants ─────────────────────────────────────────────────────────────────

const TIMEFRAMES = [
  { id: '1m', intraday: true }, { id: '5m', intraday: true },
  { id: '15m', intraday: true }, { id: '1H', intraday: true },
  { id: '1D', intraday: false }, { id: '1W', intraday: false }, { id: '1M', intraday: false },
]

const IND_COLORS = [
  '#60a5fa', '#f59e0b', '#a78bfa', '#34d399',
  '#fb923c', '#e879f9', '#2dd4bf', '#f472b6',
]

const THEME = {
  layout: { background: { color: '#161b27' }, textColor: '#94a3b8', fontSize: 11, attributionLogo: false },
  grid: { vertLines: { color: '#222a3a' }, horzLines: { color: '#222a3a' } },
  crosshair: { mode: CrosshairMode.Normal },
  rightPriceScale: { borderColor: '#2a3245' },
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function contractParams(c: Contract): string {
  return c.kind === 'continuous'
    ? `root=${encodeURIComponent(c.root ?? c.symbol)}`
    : `instrument_id=${c.instrument_id}`
}

function indicatorUrl(c: Contract, ind: { id: number | null; name: string }, tf: string): string {
  const base = `/api/explore/indicator?timeframe=${tf}`
  return c.kind === 'continuous'
    ? `${base}&name=${encodeURIComponent(ind.name)}&root=${encodeURIComponent(c.root ?? c.symbol)}`
    : `${base}&indicator_id=${ind.id}`
}

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

export default function Explore() {
  const [contracts, setContracts] = useState<Contract[]>([])
  const [selIdx, setSelIdx] = useState<number>(-1)
  const [timeframe, setTimeframe] = useState<string>('1D')
  const [bars, setBars] = useState<Bars | null>(null)
  const [available, setAvailable] = useState<IndicatorMeta[]>([])
  const [active, setActive] = useState<ActiveIndicator[]>([])
  const [loading, setLoading] = useState(false)
  const [noDb, setNoDb] = useState(false)
  const [backtests, setBacktests] = useState<BacktestMeta[]>([])
  const [overlayBtId, setOverlayBtId] = useState<number | null>(null)
  const [overlay, setOverlay] = useState<Overlay | null>(null)

  const priceRef = useRef<HTMLDivElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  const pnlRef = useRef<HTMLDivElement>(null)
  const legendRef = useRef<HTMLDivElement>(null)
  const colorCursor = useRef(0)

  const sel = selIdx >= 0 ? contracts[selIdx] : null

  // ── Load contract list once ──────────────────────────────────────────────
  useEffect(() => {
    fetch('/api/explore/contracts')
      .then(r => r.json())
      .then((d: { contracts: Contract[] }) => {
        setContracts(d.contracts ?? [])
        if (!d.contracts?.length) setNoDb(true)
        else setSelIdx(0)
      })
      .catch(() => setNoDb(true))
  }, [])

  // ── Load backtest list once (for the performance overlay selector) ────────
  useEffect(() => {
    fetch('/api/backtests')
      .then(r => r.json())
      .then((d: BacktestMeta[]) => setBacktests(Array.isArray(d) ? d : []))
      .catch(() => setBacktests([]))
  }, [])

  // ── Fetch overlay when backtest / contract / timeframe changes ────────────
  useEffect(() => {
    if (overlayBtId == null || !sel) { setOverlay(null); return }
    fetch(`/api/explore/overlay?backtest_id=${overlayBtId}&${contractParams(sel)}`)
      .then(r => r.json())
      .then((d: Overlay) => setOverlay(d))
      .catch(() => setOverlay(null))
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [overlayBtId, selIdx])

  // ── On contract change: fetch its indicators, reset active ────────────────
  useEffect(() => {
    if (!sel) return
    setActive([])
    colorCursor.current = 0
    fetch(`/api/explore/indicators?${contractParams(sel)}`)
      .then(r => r.json())
      .then((d: { indicators: IndicatorMeta[] }) => setAvailable(d.indicators ?? []))
      .catch(() => setAvailable([]))
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selIdx])

  // ── Load bars whenever contract or timeframe changes ──────────────────────
  useEffect(() => {
    if (!sel) return
    setLoading(true)
    fetch(`/api/explore/bars?${contractParams(sel)}&timeframe=${timeframe}`)
      .then(r => r.json())
      .then((d: Bars) => setBars(d))
      .catch(() => setBars({ has_data: false, candles: [], volume: [] }))
      .finally(() => setLoading(false))
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selIdx, timeframe])

  // ── Refetch active indicator data when timeframe changes (data is bucketed) ─
  useEffect(() => {
    if (!sel || active.length === 0) return
    Promise.all(
      active.map(a =>
        fetch(indicatorUrl(sel, a, timeframe))
          .then(r => r.json())
          .then((d: { data: { time: number; value: number }[] }) => ({ ...a, data: d.data ?? [] }))
      )
    ).then(setActive)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [timeframe])

  function handleAdd(meta: IndicatorMeta) {
    if (!sel || active.find(a => a.name === meta.name)) return
    const color = IND_COLORS[colorCursor.current++ % IND_COLORS.length]
    fetch(indicatorUrl(sel, meta, timeframe))
      .then(r => r.json())
      .then((d: { placement: string; data: { time: number; value: number }[] }) => {
        setActive(prev => [
          ...prev,
          { id: meta.id, name: meta.name, placement: meta.placement, color, data: d.data ?? [] },
        ])
      })
  }
  function handleRemove(name: string) { setActive(prev => prev.filter(a => a.name !== name)) }
  function handlePlacement(name: string, p: 'overlay' | 'panel') {
    setActive(prev => prev.map(a => a.name === name ? { ...a, placement: p } : a))
  }

  // ── Chart lifecycle ───────────────────────────────────────────────────────
  useEffect(() => {
    if (!bars?.has_data || !priceRef.current) return
    const priceEl = priceRef.current
    const panelEl = panelRef.current
    const intraday = TIMEFRAMES.find(t => t.id === timeframe)?.intraday ?? false

    const fmtTime = (time: number) => {
      const d = new Date(time * 1000)
      return intraday
        ? d.toLocaleString('en-US', { timeZone: 'America/New_York', month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false })
        : d.toLocaleDateString('en-US', { timeZone: 'America/New_York', month: 'short', day: 'numeric', year: '2-digit' })
    }

    const overlays = active.filter(i => i.placement === 'overlay')
    const panels = active.filter(i => i.placement === 'panel')
    const hasPnl = !!(overlay && overlay.equity.length > 0)
    const pnlEl = pnlRef.current

    const price = createChart(priceEl, {
      ...THEME,
      width: priceEl.clientWidth, height: 420,
      localization: { timeFormatter: fmtTime },
      timeScale: { borderColor: '#2a3245', timeVisible: intraday, secondsVisible: false, visible: panels.length === 0 && !hasPnl },
    })

    const candle = price.addSeries(CandlestickSeries, {
      upColor: '#4ade80', downColor: '#f87171',
      borderUpColor: '#4ade80', borderDownColor: '#f87171',
      wickUpColor: '#4ade80', wickDownColor: '#f87171',
    })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    candle.setData(bars.candles.map(c => ({ ...c, time: c.time as any })))

    // Backtest overlay: entry/exit markers on the price chart
    if (overlay?.markers.length) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      createSeriesMarkers(candle, overlay.markers.map(m => ({ ...m, time: m.time as any })) as any)
    }

    const vol = price.addSeries(HistogramSeries, { priceScaleId: 'vol', priceFormat: { type: 'volume' as const } })
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    vol.setData(bars.volume.map(v => ({ ...v, time: v.time as any })))
    price.priceScale('vol').applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } })

    for (const ind of overlays) {
      const s = price.addSeries(LineSeries, { color: ind.color, lineWidth: 2, priceLineVisible: false, lastValueVisible: true })
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      s.setData(ind.data.map(p => ({ ...p, time: p.time as any })))
    }

    let panel: ReturnType<typeof createChart> | null = null
    if (panels.length && panelEl) {
      panel = createChart(panelEl, {
        ...THEME, width: panelEl.clientWidth, height: 140,
        localization: { timeFormatter: fmtTime },
        // Hide this axis when the P&L pane sits below it (it shows its own)
        timeScale: { borderColor: '#2a3245', timeVisible: intraday, secondsVisible: false, visible: !hasPnl },
      })
      for (const ind of panels) {
        const s = panel.addSeries(LineSeries, { color: ind.color, lineWidth: 1, priceLineVisible: false, lastValueVisible: true })
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        s.setData(ind.data.map(p => ({ ...p, time: p.time as any })))
      }
    }

    // Strategy cumulative-P&L pane (baseline: green above 0, red below)
    let pnl: ReturnType<typeof createChart> | null = null
    if (hasPnl && pnlEl) {
      pnl = createChart(pnlEl, {
        ...THEME, width: pnlEl.clientWidth, height: 150,
        localization: { timeFormatter: fmtTime },
        timeScale: { borderColor: '#2a3245', timeVisible: intraday, secondsVisible: false },
      })
      const fmtPnl = (v: number) => `${v >= 0 ? '+' : ''}$${Math.abs(v).toFixed(0)}`
      const s = pnl.addSeries(BaselineSeries, {
        baseValue: { type: 'price', price: 0 },
        topLineColor: '#4ade80', topFillColor1: 'rgba(74,222,128,0.12)', topFillColor2: 'rgba(74,222,128,0.02)',
        bottomLineColor: '#f87171', bottomFillColor1: 'rgba(248,113,113,0.02)', bottomFillColor2: 'rgba(248,113,113,0.12)',
        lineWidth: 2, priceFormat: { type: 'custom' as const, minMove: 1, formatter: fmtPnl },
      })
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      s.setData(overlay!.equity.map(p => ({ ...p, time: p.time as any })))
    }

    // OHLC crosshair legend (defaults to the last bar)
    const last = bars.candles[bars.candles.length - 1]
    const idxByTime = new Map(bars.candles.map((c, i) => [c.time, i]))
    function setLegend(bar: Candle | undefined, prevClose: number | null) {
      const el = legendRef.current
      if (!el || !bar) return
      const chg = prevClose != null ? bar.close - prevClose : 0
      const pct = prevClose ? (chg / prevClose) * 100 : 0
      const cls = chg >= 0 ? '#4ade80' : '#f87171'
      const g = (k: string, v: number) => `<span><span style="color:#64748b">${k}</span> ${v}</span>`
      el.innerHTML =
        g('O', bar.open) + g('H', bar.high) + g('L', bar.low) + g('C', bar.close) +
        `<span style="color:${cls}">${chg >= 0 ? '+' : ''}${chg.toFixed(2)} (${chg >= 0 ? '+' : ''}${pct.toFixed(2)}%)</span>`
    }
    const lastPrev = bars.candles[bars.candles.length - 2]
    setLegend(last, lastPrev ? lastPrev.close : null)
    price.subscribeCrosshairMove(param => {
      const bar = param.seriesData?.get(candle) as Candle | undefined
      if (!bar || param.time == null) { setLegend(last, lastPrev ? lastPrev.close : null); return }
      const i = idxByTime.get(param.time as number) ?? 0
      setLegend(bar, i > 0 ? bars.candles[i - 1].close : null)
    })

    // Sync time scales across price + indicator panel + P&L pane
    const charts = [price, ...(panel ? [panel] : []), ...(pnl ? [pnl] : [])]
    let syncing = false
    for (const src of charts) {
      src.timeScale().subscribeVisibleLogicalRangeChange(r => {
        if (syncing || !r) return
        syncing = true
        for (const tgt of charts) if (tgt !== src) tgt.timeScale().setVisibleLogicalRange(r)
        syncing = false
      })
    }
    price.timeScale().fitContent()
    panel?.timeScale().fitContent()
    pnl?.timeScale().fitContent()

    const ro = new ResizeObserver(() => {
      price.applyOptions({ width: priceEl.clientWidth })
      if (panel && panelEl) panel.applyOptions({ width: panelEl.clientWidth })
      if (pnl && pnlEl) pnl.applyOptions({ width: pnlEl.clientWidth })
    })
    ro.observe(priceEl)

    return () => { ro.disconnect(); price.remove(); panel?.remove(); pnl?.remove() }
  }, [bars, active, timeframe, overlay])

  const overlays = active.filter(i => i.placement === 'overlay')
  const panels = active.filter(i => i.placement === 'panel')
  const hasChart = bars?.has_data ?? false

  return (
    <div data-bs-theme="dark" style={{ minHeight: '100vh', background: '#0f1117' }}>
      <Navbar />
      <div className="container-fluid py-3 px-4" style={{ maxWidth: 1400 }}>

        <nav className="btk-breadcrumb mb-3" aria-label="breadcrumb">
          <ol className="breadcrumb mb-0">
            <li className="breadcrumb-item"><Link to="/">Home</Link></li>
            <li className="breadcrumb-item active">Chart Explorer</li>
          </ol>
        </nav>

        {noDb ? (
          <div className="btk-chart-card d-flex flex-column align-items-center justify-content-center" style={{ padding: '48px 24px', gap: 8, textAlign: 'center' }}>
            <i className="bi bi-bar-chart-steps" style={{ fontSize: '2rem', color: 'var(--btk-muted-dk)', opacity: 0.4 }} />
            <div style={{ color: 'var(--btk-muted-dk)', fontSize: '0.9rem', fontWeight: 500 }}>No market data available</div>
            <div style={{ color: 'var(--btk-muted-dk)', fontSize: '0.78rem' }}>
              Start the server with <code style={{ color: '#93c5fd' }}>BTKIT_INPUT_DB=…</code> to browse contracts.
            </div>
          </div>
        ) : (
          <>
            {/* Controls */}
            <div className="d-flex align-items-center flex-wrap mb-2" style={{ gap: 8 }}>
              <select
                aria-label="Contract"
                value={selIdx}
                onChange={e => setSelIdx(parseInt(e.target.value))}
                style={{ background: '#1e2535', border: '1px solid #2a3245', borderRadius: 6, color: '#cbd5e1', fontSize: '0.8rem', fontFamily: 'ui-monospace, monospace', height: 32, padding: '0 10px', cursor: 'pointer' }}
              >
                {contracts.map((c, i) => (
                  <option key={i} value={i}>
                    {c.symbol}{c.kind === 'continuous' ? ' · front-month' : ''} — {c.start} → {c.end}
                  </option>
                ))}
              </select>

              <div style={{ width: 1, height: 24, background: '#2a3245' }} />

              {/* Timeframe segmented control */}
              <div className="d-flex" style={{ background: '#1e2535', borderRadius: 6, padding: 2, gap: 2 }}>
                {TIMEFRAMES.map((t, i) => (
                  <div key={t.id} style={{ display: 'flex' }}>
                    {i > 0 && t.intraday !== TIMEFRAMES[i - 1].intraday && (
                      <div style={{ width: 1, background: '#2a3245', margin: '4px 3px' }} />
                    )}
                    <button
                      onClick={() => setTimeframe(t.id)}
                      style={{
                        background: timeframe === t.id ? '#2563eb' : 'transparent',
                        border: 'none', borderRadius: 4, padding: '4px 9px',
                        color: timeframe === t.id ? '#fff' : '#64748b',
                        cursor: 'pointer', fontSize: '0.72rem', fontWeight: 600, fontFamily: 'ui-monospace, monospace',
                      }}
                    >{t.id}</button>
                  </div>
                ))}
              </div>

              {available.length > 0 && (
                <>
                  <div style={{ width: 1, height: 24, background: '#2a3245' }} />
                  <select
                    aria-label="Add indicator"
                    value=""
                    onChange={e => {
                      const meta = available.find(a => a.name === e.target.value)
                      if (meta) handleAdd(meta)
                      e.currentTarget.value = ''
                    }}
                    style={{ background: '#1e2535', border: '1px solid #2a3245', borderRadius: 6, color: '#94a3b8', fontSize: '0.75rem', height: 30, padding: '0 8px', cursor: 'pointer' }}
                  >
                    <option value="">+ Indicator…</option>
                    {available.filter(a => !active.find(i => i.name === a.name)).map(a => (
                      <option key={a.name} value={a.name}>{a.name}</option>
                    ))}
                  </select>

                  {active.map(ind => (
                    <div key={ind.name} style={{ display: 'flex', alignItems: 'center', gap: 5, background: '#1e2535', borderRadius: 6, padding: '3px 6px 3px 8px', border: '1px solid #2a3245' }}>
                      <span style={{ width: 7, height: 7, borderRadius: '50%', background: ind.color, flexShrink: 0 }} />
                      <span style={{ fontSize: '0.72rem', color: '#94a3b8', fontFamily: 'ui-monospace, monospace' }}>{ind.name}</span>
                      <div style={{ display: 'flex', background: '#0f1117', borderRadius: 4, padding: 1, gap: 1, marginLeft: 2 }}>
                        {(['overlay', 'panel'] as const).map(p => (
                          <button key={p} onClick={() => handlePlacement(ind.name, p)}
                            title={p === 'overlay' ? 'Overlay on price' : 'Separate panel'}
                            style={{ background: ind.placement === p ? '#2563eb' : 'transparent', border: 'none', borderRadius: 3, padding: '2px 5px', color: ind.placement === p ? '#fff' : '#475569', cursor: 'pointer', fontSize: '0.65rem', fontWeight: 600, lineHeight: 1.4 }}
                          >{p === 'overlay' ? '⬆' : '⬇'}</button>
                        ))}
                      </div>
                      <button onClick={() => handleRemove(ind.name)} title="Remove indicator"
                        style={{ background: 'none', border: 'none', color: '#475569', cursor: 'pointer', padding: '0 2px', fontSize: '0.85rem', lineHeight: 1 }}>×</button>
                    </div>
                  ))}
                </>
              )}

              {/* Backtest performance overlay */}
              {backtests.length > 0 && (
                <>
                  <div style={{ width: 1, height: 24, background: '#2a3245' }} />
                  <select
                    aria-label="Overlay backtest"
                    value={overlayBtId ?? ''}
                    onChange={e => setOverlayBtId(e.target.value ? parseInt(e.target.value) : null)}
                    style={{ background: '#1e2535', border: '1px solid #2a3245', borderRadius: 6, color: overlayBtId == null ? '#64748b' : '#93c5fd', fontSize: '0.75rem', height: 30, padding: '0 8px', cursor: 'pointer', maxWidth: 240 }}
                  >
                    <option value="">Overlay backtest…</option>
                    {backtests.map(b => (
                      <option key={b.id} value={b.id}>#{b.id} · {b.strategy_label || b.strategy_name}</option>
                    ))}
                  </select>
                  {overlayBtId != null && overlay && (
                    <span style={{ fontSize: '0.72rem', color: '#64748b', fontFamily: 'ui-monospace, monospace' }}>
                      {overlay.n_positions} trade{overlay.n_positions === 1 ? '' : 's'} on {sel?.symbol}
                    </span>
                  )}
                </>
              )}
            </div>

            {/* Chart card */}
            <div className="btk-chart-card mb-4" style={{ position: 'relative' }}>
              {loading && (
                <div style={{ position: 'absolute', inset: 0, zIndex: 10, borderRadius: 10, background: 'rgba(22,27,39,0.55)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <i className="bi bi-arrow-repeat" style={{ animation: 'spin 1.2s linear infinite', display: 'inline-block', fontSize: '1.5rem', color: '#64748b' }} />
                </div>
              )}

              <div className="d-flex align-items-center px-3 pt-3 pb-2" style={{ gap: 14 }}>
                <span className="btk-chart-title" style={{ fontFamily: 'ui-monospace, monospace', color: '#e2e8f0' }}>
                  {sel?.symbol ?? '—'}
                </span>
                <span style={{ fontFamily: 'ui-monospace, monospace', fontSize: '0.7rem', color: '#64748b', background: '#1e2535', borderRadius: 5, padding: '2px 7px' }}>{timeframe}</span>
                <div ref={legendRef} className="d-flex" style={{ gap: 14, fontFamily: 'ui-monospace, monospace', fontSize: '0.75rem', color: '#94a3b8', fontVariantNumeric: 'tabular-nums' }} />
                <div className="d-flex" style={{ marginLeft: 'auto', gap: 12, fontFamily: 'ui-monospace, monospace', fontSize: '0.72rem' }}>
                  {overlays.map(i => <span key={i.name} style={{ color: i.color }}>{i.name}</span>)}
                </div>
              </div>

              {!hasChart && !loading ? (
                <div className="d-flex flex-column align-items-center justify-content-center" style={{ padding: '48px 24px', gap: 8, color: 'var(--btk-muted-dk)', fontSize: '0.85rem' }}>
                  No bars for this contract / timeframe.
                </div>
              ) : (
                <>
                  <div ref={priceRef} />
                  <div style={{ display: panels.length > 0 ? 'block' : 'none', borderTop: '1px solid #2a3245' }}>
                    <div className="d-flex align-items-center gap-3 px-3 pt-2 pb-1" style={{ fontSize: '0.68rem', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: '#64748b', fontFamily: 'ui-monospace, monospace' }}>
                      {panels.map(i => <span key={i.name} style={{ color: i.color }}>— {i.name}</span>)}
                    </div>
                    <div ref={panelRef} />
                  </div>
                  <div style={{ display: overlay && overlay.equity.length > 0 ? 'block' : 'none', borderTop: '1px solid #2a3245' }}>
                    <div className="px-3 pt-2 pb-1" style={{ fontSize: '0.68rem', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: '#64748b', fontFamily: 'ui-monospace, monospace' }}>
                      Cumulative P&amp;L
                    </div>
                    <div ref={pnlRef} />
                  </div>
                </>
              )}
            </div>
          </>
        )}
      </div>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  )
}
