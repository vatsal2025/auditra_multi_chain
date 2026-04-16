import { useEffect, useState } from 'react'

export type ToastType = 'success' | 'error' | 'info'

export interface ToastData {
  id: string
  message: string
  type: ToastType
}

interface Props {
  toasts: ToastData[]
  onDismiss: (id: string) => void
}

export default function ToastContainer({ toasts, onDismiss }: Props) {
  return (
    <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-2">
      {toasts.map(t => (
        <Toast key={t.id} toast={t} onDismiss={onDismiss} />
      ))}
    </div>
  )
}

function Toast({ toast, onDismiss }: { toast: ToastData; onDismiss: (id: string) => void }) {
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    requestAnimationFrame(() => setVisible(true))
    const timer = setTimeout(() => {
      setVisible(false)
      setTimeout(() => onDismiss(toast.id), 300)
    }, 4000)
    return () => clearTimeout(timer)
  }, [toast.id, onDismiss])

  const bg = {
    success: 'bg-green-900/90 border-green-700 text-green-200',
    error: 'bg-red-900/90 border-red-700 text-red-200',
    info: 'bg-slate-800/90 border-slate-600 text-slate-200',
  }[toast.type]

  return (
    <div
      onClick={() => onDismiss(toast.id)}
      className={`px-4 py-3 rounded-xl border text-sm font-medium cursor-pointer
        max-w-sm shadow-xl transition-all duration-300
        ${visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}
        ${bg}`}
    >
      {toast.message}
    </div>
  )
}

let _addToast: ((msg: string, type: ToastType) => void) | null = null

export function useToast() {
  const [toasts, setToasts] = useState<ToastData[]>([])

  _addToast = (message: string, type: ToastType = 'info') => {
    const id = Math.random().toString(36).slice(2)
    setToasts(prev => [...prev, { id, message, type }])
  }

  const dismiss = (id: string) => setToasts(prev => prev.filter(t => t.id !== id))

  return { toasts, dismiss, toast: _addToast }
}
