import { useRef, useEffect } from 'react'
import type { CSSProperties } from 'react'

// Plotly is loaded as a standalone <script> in index.html — not bundled.
declare const Plotly: {
  newPlot: (el: HTMLElement, data: unknown[], layout: unknown, config: unknown) => void
  react:   (el: HTMLElement, data: unknown[], layout: unknown, config: unknown) => void
  purge:   (el: HTMLElement) => void
}

const CONFIG = { displayModeBar: false, responsive: true }

interface Props {
  data:    unknown[]
  layout:  unknown
  style?:  CSSProperties
}

export default function PlotlyChart({ data, layout, style }: Props) {
  const el        = useRef<HTMLDivElement>(null)
  const didMount  = useRef(false)

  useEffect(() => {
    if (el.current) Plotly.newPlot(el.current, data, layout, CONFIG)
    didMount.current = true
    return () => {
      didMount.current = false
      if (el.current) Plotly.purge(el.current)
    }
  }, [])  // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    // Skip the first run — newPlot already handled the initial render.
    if (!didMount.current) return
    if (el.current) Plotly.react(el.current, data, layout, CONFIG)
  }, [data, layout])

  return <div ref={el} style={style} />
}
