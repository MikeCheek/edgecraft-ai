// Browser-side audio helpers.
//
// MediaRecorder produces WebM/Opus (Chrome, Firefox) or MP4/AAC (Safari),
// which the backend can only decode when ffmpeg is installed. Converting to
// 16 kHz mono 16-bit WAV in the browser makes recordings work everywhere and
// matches the sample rate the audio models are trained on.

export const TARGET_SAMPLE_RATE = 16000

export function encodeWav (samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2)
  const view = new DataView(buffer)
  const writeStr = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i))
  }
  writeStr(0, 'RIFF')
  view.setUint32(4, 36 + samples.length * 2, true)
  writeStr(8, 'WAVE')
  writeStr(12, 'fmt ')
  view.setUint32(16, 16, true) // PCM chunk size
  view.setUint16(20, 1, true) // PCM
  view.setUint16(22, 1, true) // mono
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true) // byte rate
  view.setUint16(32, 2, true) // block align
  view.setUint16(34, 16, true) // bits per sample
  writeStr(36, 'data')
  view.setUint32(40, samples.length * 2, true)
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true)
  }
  return new Blob([buffer], { type: 'audio/wav' })
}

/** Decode any browser-recordable audio blob to a 16 kHz mono WAV blob. */
export async function blobToWav16k (blob: Blob): Promise<Blob> {
  const AudioCtx: typeof AudioContext =
    window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
  const ctx = new AudioCtx()
  try {
    const decoded = await ctx.decodeAudioData(await blob.arrayBuffer())
    const length = Math.max(1, Math.ceil(decoded.duration * TARGET_SAMPLE_RATE))
    const offline = new OfflineAudioContext(1, length, TARGET_SAMPLE_RATE)
    const source = offline.createBufferSource()
    source.buffer = decoded // multi-channel input is down-mixed to mono
    source.connect(offline.destination)
    source.start()
    const rendered = await offline.startRendering()
    return encodeWav(rendered.getChannelData(0), TARGET_SAMPLE_RATE)
  } finally {
    ctx.close().catch(() => {})
  }
}

/** Decode any audio blob to `nSamples` of 16 kHz mono int16 PCM (padded or
 *  trimmed) - the format the exported sketches' Serial harness expects. */
export async function blobToPcm16 (blob: Blob, nSamples: number): Promise<Int16Array> {
  const wav = await blobToWav16k(blob)
  const view = new DataView(await wav.arrayBuffer())
  const available = (view.byteLength - 44) / 2
  const out = new Int16Array(nSamples)
  for (let i = 0; i < Math.min(nSamples, available); i++) out[i] = view.getInt16(44 + i * 2, true)
  return out
}
