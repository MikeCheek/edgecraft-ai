import { describe, expect, it } from 'vitest'
import { parseDeviceLine } from '../components/DeviceMonitor'

describe('parseDeviceLine', () => {
  it('extracts latency, arena use and predictions printed by the sketch', () => {
    let s = { latenciesUs: [] as number[] }
    for (const line of [
      'Tensor arena used: 24576 of 40960 bytes',
      'Inference: 1834 us',
      'MFCC: 900 us, inference: 2100 us',
      'Best: yes  (97.12%)',
    ]) s = parseDeviceLine(line, s)
    expect(s.latenciesUs).toEqual([1834, 2100])
    expect(s).toMatchObject({ arenaUsed: 24576, arenaTotal: 40960, lastPrediction: 'yes  (97.12%)' })
    expect(parseDeviceLine('3 object(s)', s).lastPrediction).toBe('3 object(s)')
  })
})
