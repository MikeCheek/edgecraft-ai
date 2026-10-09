import { describe, expect, it } from 'vitest'
import { encodeWav } from '../utils/audio'

describe('encodeWav', () => {
  it('writes a valid 16-bit mono PCM header and clamps samples', async () => {
    const samples = new Float32Array([0, 0.5, -0.5, 2, -2])
    const blob = encodeWav(samples, 16000)
    const view = new DataView(await blob.arrayBuffer())
    const tag = (o: number) => String.fromCharCode(...[0, 1, 2, 3].map(i => view.getUint8(o + i)))
    expect(tag(0)).toBe('RIFF')
    expect(tag(8)).toBe('WAVE')
    expect(view.getUint16(22, true)).toBe(1) // mono
    expect(view.getUint32(24, true)).toBe(16000)
    expect(view.getUint16(34, true)).toBe(16)
    expect(view.getUint32(40, true)).toBe(samples.length * 2)
    expect(view.getInt16(44 + 3 * 2, true)).toBe(32767) // clamped +2
    expect(view.getInt16(44 + 4 * 2, true)).toBe(-32768) // clamped -2
  })
})
