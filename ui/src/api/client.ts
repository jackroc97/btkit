// Typed API client — generated from /openapi.json via openapi-typescript once
// the FastAPI backend is wired up (Phase 1). Hand-written stubs until then.

const BASE = '/api'

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json() as Promise<T>
}

export const api = { get }
