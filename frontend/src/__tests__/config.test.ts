import { beforeEach, describe, expect, it, vi } from 'vitest'

const store: Record<string, string> = {}
vi.stubGlobal('localStorage', {
  getItem: (k: string) => store[k] ?? null,
  setItem: (k: string, v: string) => { store[k] = v },
  removeItem: (k: string) => { delete store[k] },
})

describe('config', () => {
  beforeEach(() => { Object.keys(store).forEach(k => delete store[k]) })

  it('adds the API key to direct URLs only when one is set', async () => {
    const { withAuthQuery, saveConnection, authHeaders } = await import('../config')
    expect(withAuthQuery('/api/x')).toBe('/api/x')
    expect(authHeaders()).toEqual({})
    saveConnection('', 'k e y')
    expect(withAuthQuery('/api/x')).toBe('/api/x?api_key=k%20e%20y')
    expect(withAuthQuery('/api/x?a=1')).toBe('/api/x?a=1&api_key=k%20e%20y')
    expect(authHeaders()).toEqual({ 'X-API-Key': 'k e y' })
  })

  it('defaults the API base to the local backend', async () => {
    const { API_BASE, WS_BASE } = await import('../config')
    expect(API_BASE).toBe('http://localhost:8000/api')
    expect(WS_BASE).toBe('ws://localhost:8000')
  })
})
