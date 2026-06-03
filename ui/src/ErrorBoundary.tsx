import { Component } from 'react'
import type { ReactNode } from 'react'

interface Props { children: ReactNode }
interface State { error: Error | null }

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  render() {
    const { error } = this.state
    if (error) {
      return (
        <div style={{ padding: 32, fontFamily: 'monospace', color: '#f87171', background: '#0f1117', minHeight: '100vh' }}>
          <div style={{ marginBottom: 8, fontSize: 14, fontWeight: 700 }}>Runtime error</div>
          <pre style={{ whiteSpace: 'pre-wrap', fontSize: 12 }}>{error.message}</pre>
          <pre style={{ whiteSpace: 'pre-wrap', fontSize: 11, color: '#64748b', marginTop: 16 }}>{error.stack}</pre>
        </div>
      )
    }
    return this.props.children
  }
}
