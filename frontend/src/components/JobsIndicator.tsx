// Header pill showing the job the ML worker is busy with (and how many are
// waiting), so long trainings/optimizations stay visible from every page.

import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Loader, Clock } from 'lucide-react'
import { apiFetch } from '../config'

interface ActiveJob {
  kind: 'training' | 'optimization'
  id: string
  label: string
  progress?: number
  status: string
}

export function JobsIndicator ({ enabled }: { enabled: boolean }) {
  const navigate = useNavigate()
  const [job, setJob] = useState<ActiveJob | null>(null)
  const [pending, setPending] = useState(0)

  useEffect(() => {
    if (!enabled) return
    let alive = true
    const poll = async () => {
      try {
        const [t, o, q] = await Promise.all([
          apiFetch('/training/active').then(r => r.json()),
          apiFetch('/optimization/active').then(r => r.json()),
          apiFetch('/training/queue').then(r => r.json()),
        ])
        if (!alive) return
        const running = [t?.session && { ...t.session, kind: 'training' }, o?.session && { ...o.session, kind: 'optimization' }]
          .filter(Boolean)
          .sort((a, b) => (a.status === 'running' ? -1 : 0) - (b.status === 'running' ? -1 : 0))[0]
        setJob(running ? {
          kind: running.kind,
          id: running.id,
          status: running.status,
          progress: running.progress,
          label: running.kind === 'training'
            ? `${running.name || running.base_model} ${running.status === 'running' ? `${running.current_epoch}/${running.total_epochs}` : ''}`
            : `${(running.frontend_method || running.method || '').replace(/_/g, ' ').toLowerCase()}`,
        } : null)
        setPending(q?.pending?.length ?? 0)
      } catch { /* backend offline - the health badge already says so */ }
    }
    poll()
    const t = setInterval(poll, 5000)
    return () => { alive = false; clearInterval(t) }
  }, [enabled])

  if (!job && pending === 0) return null
  return (
    <button
      onClick={() => navigate(job?.kind === 'optimization' ? '/optimize' : '/train')}
      className="hidden md:flex items-center gap-2 px-3 py-1.5 bg-violet-500/10 border border-violet-500/30 rounded-full text-xs text-violet-200 hover:bg-violet-500/20 max-w-xs"
      title="Background ML jobs"
    >
      {job ? <Loader className="w-3 h-3 animate-spin shrink-0" /> : <Clock className="w-3 h-3 shrink-0" />}
      <span className="truncate">
        {job ? `${job.kind === 'training' ? 'Training' : 'Optimizing'}: ${job.label}` : 'Jobs waiting'}
      </span>
      {pending > 0 && <span className="px-1.5 rounded-full bg-violet-600 text-white">+{pending}</span>}
    </button>
  )
}
