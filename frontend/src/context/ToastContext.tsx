import React, { createContext, useContext, useState, useCallback, useRef } from 'react'
import { createPortal } from 'react-dom'
import { CheckCircle, XCircle, AlertTriangle, Info, X } from 'lucide-react'

export type ToastType = 'success' | 'error' | 'warning' | 'info'

interface Toast {
  id: string
  type: ToastType
  message: string
  duration?: number
}

interface ToastContextValue {
  toast: (type: ToastType, message: string, duration?: number) => void
}

const ToastContext = createContext<ToastContextValue | null>(null)

const ICONS: Record<ToastType, React.ReactNode> = {
  success: <CheckCircle className="w-4 h-4 text-green-400 shrink-0" />,
  error: <XCircle className="w-4 h-4 text-red-400 shrink-0" />,
  warning: <AlertTriangle className="w-4 h-4 text-yellow-400 shrink-0" />,
  info: <Info className="w-4 h-4 text-blue-400 shrink-0" />,
}

const BG_COLORS: Record<ToastType, string> = {
  success: 'border-green-500/30 bg-green-500/10',
  error: 'border-red-500/30 bg-red-500/10',
  warning: 'border-yellow-500/30 bg-yellow-500/10',
  info: 'border-blue-500/30 bg-blue-500/10',
}

let _counter = 0

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([])
  const timersRef = useRef<Map<string, number>>(new Map())

  const removeToast = useCallback((id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id))
    const timer = timersRef.current.get(id)
    if (timer) {
      clearTimeout(timer)
      timersRef.current.delete(id)
    }
  }, [])

  const toast = useCallback((type: ToastType, message: string, duration = 4000) => {
    const id = `toast-${++_counter}`
    setToasts(prev => [...prev, { id, type, message, duration }])

    if (duration > 0) {
      const timer = window.setTimeout(() => removeToast(id), duration)
      timersRef.current.set(id, timer)
    }
  }, [removeToast])

  return (
    <ToastContext.Provider value={{ toast }}>
      {children}
      {createPortal(
        <div
          className="fixed bottom-4 right-4 z-[9999] flex flex-col gap-2 max-w-sm"
          role="region"
          aria-label="Notifications"
        >
          {toasts.map(t => (
            <div
              key={t.id}
              className={`flex items-start gap-3 px-4 py-3 rounded-xl border backdrop-blur-xl shadow-2xl animate-slideIn ${BG_COLORS[t.type]}`}
              role="alert"
            >
              {ICONS[t.type]}
              <p className="text-sm text-gray-200 flex-1 break-words">{t.message}</p>
              <button
                onClick={() => removeToast(t.id)}
                className="text-gray-500 hover:text-gray-300 shrink-0 mt-0.5"
                aria-label="Dismiss"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          ))}
        </div>,
        document.body
      )}
    </ToastContext.Provider>
  )
}

export function useToast(): ToastContextValue {
  const ctx = useContext(ToastContext)
  if (!ctx) throw new Error('useToast must be used within a ToastProvider')
  return ctx
}
