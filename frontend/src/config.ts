// Central backend connection settings.
//
// Resolution order for the API base URL:
//   1. a value saved from the Settings page (localStorage)
//   2. VITE_API_BASE_URL at build time (Docker builds use "/api", proxied by nginx)
//   3. http://localhost:8000/api (local dev with `uvicorn` + `vite`)
// The optional API key (backend EDGECRAFT_API_TOKEN) is stored per browser.

const API_BASE_KEY = 'ec_api_base'
const API_KEY_KEY = 'ec_api_key'

function readStorage (key: string): string | null {
  try {
    return localStorage.getItem(key)
  } catch {
    return null
  }
}

function resolveApiBase (): string {
  const saved = readStorage(API_BASE_KEY)
  const fromEnv = import.meta.env.VITE_API_BASE_URL as string | undefined
  return (saved || fromEnv || 'http://localhost:8000/api').replace(/\/+$/, '')
}

export const API_BASE = resolveApiBase()

/** ws(s)://host[/prefix] matching API_BASE, without the trailing /api. */
export const WS_BASE = (() => {
  const base = API_BASE.replace(/\/api$/, '')
  if (/^https?:\/\//.test(base)) return base.replace(/^http/, 'ws')
  const proto = typeof window !== 'undefined' && window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const host = typeof window !== 'undefined' ? window.location.host : 'localhost'
  return `${proto}//${host}${base}`
})()

export function getApiKey (): string {
  return readStorage(API_KEY_KEY) || ''
}

export function getSavedApiBase (): string {
  return readStorage(API_BASE_KEY) || ''
}

/** Persist connection settings; callers reload the page to apply them. */
export function saveConnection (apiBase: string, apiKey: string): void {
  try {
    if (apiBase.trim()) localStorage.setItem(API_BASE_KEY, apiBase.trim())
    else localStorage.removeItem(API_BASE_KEY)
    if (apiKey.trim()) localStorage.setItem(API_KEY_KEY, apiKey.trim())
    else localStorage.removeItem(API_KEY_KEY)
  } catch {
    /* storage unavailable - settings apply for this page view only */
  }
}

export function authHeaders (): Record<string, string> {
  const key = getApiKey()
  return key ? { 'X-API-Key': key } : {}
}

/** Adds ?api_key= for URLs the browser loads directly (img src, downloads,
 *  EventSource, WebSocket), which can't carry custom headers. */
export function withAuthQuery (url: string): string {
  const key = getApiKey()
  if (!key) return url
  return `${url}${url.includes('?') ? '&' : '?'}api_key=${encodeURIComponent(key)}`
}

/** fetch() against the backend with the API key attached. `path` may be a
 *  full URL or a path relative to API_BASE (starting with "/"). */
export function apiFetch (path: string, init: RequestInit = {}): Promise<Response> {
  const url = path.startsWith('/') ? `${API_BASE}${path}` : path
  const headers = new Headers(init.headers)
  Object.entries(authHeaders()).forEach(([k, v]) => headers.set(k, v))
  return fetch(url, { ...init, headers })
}
