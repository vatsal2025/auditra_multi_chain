import { useCallback, useEffect, useRef, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import axios from 'axios'
import {
  AuditResponse,
  ColumnInfo,
  runAudit,
  UploadResponse,
  uploadDataset,
} from '../services/api'

interface Props {
  onUploadComplete: (data: UploadResponse) => void
  onAuditComplete: (data: AuditResponse) => void
  uploadData: UploadResponse | null
  auditData: AuditResponse | null
}

const DEMO_STEPS = [
  { label: 'Loading 48,842 rows…',              ms: 900 },
  { label: 'Detecting column types…',            ms: 800 },
  { label: 'Building feature correlation graph…', ms: 1200 },
  { label: 'Tracing relay chains (depth 4)…',    ms: 2000 },
  { label: 'Scoring paths with Vertex AI (AutoML)…', ms: 1800 },
  { label: 'Computing fairness metrics…',         ms: 1500 },
  { label: 'Running Vertex AI outcome scorer…',   ms: 1200 },
  { label: 'Generating audit report…',            ms: 600 },
]
const DEMO_TOTAL_MS = DEMO_STEPS.reduce((s, x) => s + x.ms, 0)

function DemoProgress({ onDone, waiting }: { onDone: () => void; waiting?: boolean }) {
  const [step, setStep] = useState(0)
  const [pct, setPct] = useState(0)
  const startRef = useRef(Date.now())
  const doneRef = useRef(false)

  useEffect(() => {
    let elapsed = 0
    const timers: ReturnType<typeof setTimeout>[] = []
    DEMO_STEPS.forEach((s, i) => {
      timers.push(setTimeout(() => setStep(i), elapsed))
      elapsed += s.ms
    })
    // smooth progress bar
    const interval = setInterval(() => {
      const t = Date.now() - startRef.current
      setPct(Math.min(99, (t / DEMO_TOTAL_MS) * 100))
    }, 80)
    timers.push(setTimeout(() => {
      clearInterval(interval)
      setPct(100)
      if (!doneRef.current) { doneRef.current = true; onDone() }
    }, DEMO_TOTAL_MS))
    return () => { timers.forEach(clearTimeout); clearInterval(interval) }
  }, [onDone])

  return (
    <div className="mb-6 p-6 bg-slate-800/80 border border-slate-600 rounded-xl space-y-4">
      <div className="flex items-center gap-3">
        <div className="w-5 h-5 border-2 border-red-400 border-t-transparent rounded-full animate-spin shrink-0" />
        <span className="text-white font-semibold text-sm">
          {waiting ? 'Finalising results…' : 'Auditing Adult Income dataset…'}
        </span>
        <span className="ml-auto text-red-400 font-mono text-sm font-bold">{Math.round(pct)}%</span>
      </div>

      {/* Progress bar */}
      <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-red-600 to-orange-400 rounded-full transition-all"
          style={{ width: `${pct}%`, transition: 'width 0.08s linear' }}
        />
      </div>

      {/* Steps */}
      <div className="space-y-1.5">
        {DEMO_STEPS.map((s, i) => (
          <div key={s.label} className={`flex items-center gap-2 text-xs transition-all duration-300
            ${i < step ? 'text-green-400' : i === step ? 'text-white' : 'text-slate-600'}`}>
            {i < step
              ? <span className="w-3.5 shrink-0">✓</span>
              : i === step
                ? <span className="w-3.5 h-3.5 border border-red-400 border-t-transparent rounded-full animate-spin shrink-0 inline-block" />
                : <span className="w-3.5 shrink-0 text-slate-700">○</span>
            }
            {s.label}
          </div>
        ))}
      </div>
    </div>
  )
}

export default function UploadScreen({ onUploadComplete, onAuditComplete, uploadData }: Props) {
  const [uploading, setUploading] = useState(false)
  const [auditing, setAuditing] = useState(false)
  const [demoLoading, setDemoLoading] = useState(false)
  const [demoReady, setDemoReady] = useState<{ upload: UploadResponse; audit: AuditResponse } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [columns, setColumns] = useState<ColumnInfo[]>([])
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [maxDepth, setMaxDepth] = useState(4)
  const [rowCount, setRowCount] = useState(0)

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (!file) return
    setError(null)
    setUploading(true)
    try {
      const res = await uploadDataset(file)
      setColumns(res.data.columns)
      setSessionId(res.data.session_id)
      setRowCount(res.data.row_count)
      onUploadComplete(res.data)
    } catch (e: any) {
      setError(e.response?.data?.detail || 'Upload failed')
    } finally {
      setUploading(false)
    }
  }, [onUploadComplete])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'] },
    maxFiles: 1,
  })

  const toggleCol = (col: string) => {
    setSelected(prev => {
      const s = new Set(prev)
      s.has(col) ? s.delete(col) : s.add(col)
      return s
    })
  }

  const handleAudit = async () => {
    if (!sessionId || selected.size === 0) return
    setAuditing(true)
    setError(null)
    try {
      const res = await runAudit(sessionId, Array.from(selected), maxDepth)
      onAuditComplete(res.data)
    } catch (e: any) {
      setError(e.response?.data?.detail || 'Audit failed')
    } finally {
      setAuditing(false)
    }
  }

  const animDoneRef = useRef(false)
  const dataReadyRef = useRef<{ upload: UploadResponse; audit: AuditResponse } | null>(null)

  const revealDemo = (data: { upload: UploadResponse; audit: AuditResponse }) => {
    onUploadComplete(data.upload)
    onAuditComplete(data.audit)
    setDemoLoading(false)
  }

  const handleDemo = async () => {
    animDoneRef.current = false
    dataReadyRef.current = null
    setDemoLoading(true)
    setDemoReady(null)
    setError(null)
    try {
      const res = await axios.post<{ upload: UploadResponse; audit: AuditResponse }>('/api/demo/adult')
      dataReadyRef.current = res.data
      setDemoReady(res.data)
      // If animation already finished, reveal immediately
      if (animDoneRef.current) revealDemo(res.data)
    } catch (e: any) {
      setDemoLoading(false)
      setError(e.response?.data?.detail || 'Could not load demo.')
    }
  }

  const handleDemoAnimationDone = () => {
    animDoneRef.current = true
    // If API already returned, reveal immediately; otherwise wait for API
    if (dataReadyRef.current) revealDemo(dataReadyRef.current)
    // else: API response handler above will call revealDemo when it arrives
  }

  return (
    <div className="max-w-3xl mx-auto">
      {/* Hero */}
      <div className="text-center mb-10">
        <div className="inline-flex items-center gap-2 bg-red-500/10 border border-red-500/30 rounded-full px-4 py-1.5 text-red-400 text-xs font-semibold uppercase tracking-widest mb-6">
          EU AI Act Article 10 Compliance
        </div>
        <h1 className="text-5xl font-bold text-white mb-4 leading-tight">
          Find Hidden<br />
          <span className="text-red-400">Discrimination Chains</span>
        </h1>
        <p className="text-slate-400 text-lg max-w-xl mx-auto">
          Existing tools catch <code className="text-slate-300 bg-slate-800 px-1 rounded">zip code → race</code>.
          We catch <code className="text-slate-300 bg-slate-800 px-1 rounded">zip → income → credit → race</code>.
          The 3-hop chains every biased AI is built on.
        </p>
      </div>

      {/* Demo banner / progress */}
      {!columns.length && !demoLoading && (
        <div className="mb-6 p-4 bg-slate-800/60 border border-slate-600 rounded-xl flex items-center justify-between gap-4">
          <div>
            <p className="text-white text-sm font-semibold">Try with Adult Income — see HIGH-risk chains in seconds</p>
            <p className="text-slate-400 text-xs mt-0.5">
              UCI dataset behind Amazon's 2018 hiring AI scandal — occupation → sex chains, skill 0.51
            </p>
          </div>
          <button
            onClick={handleDemo}
            className="shrink-0 px-5 py-2 bg-red-600 hover:bg-red-700 text-white text-sm font-bold rounded-lg transition-colors whitespace-nowrap"
          >
            Run Demo →
          </button>
        </div>
      )}
      {!columns.length && demoLoading && (
        <DemoProgress onDone={handleDemoAnimationDone} waiting={animDoneRef.current && !demoReady} />
      )}

      {/* Dropzone */}
      {!columns.length && (
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-xl p-16 text-center cursor-pointer transition-all
            ${isDragActive ? 'border-red-400 bg-red-400/10 scale-[1.01]' : 'border-slate-600 hover:border-slate-400 hover:bg-slate-800/30'}`}
        >
          <input {...getInputProps()} />
          {uploading ? (
            <div>
              <div className="w-8 h-8 border-2 border-red-500 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
              <p className="text-slate-300">Uploading…</p>
            </div>
          ) : isDragActive ? (
            <p className="text-red-300 text-lg">Drop it here</p>
          ) : (
            <>
              <div className="text-4xl mb-4">📊</div>
              <p className="text-slate-300 text-lg mb-2">Drop your CSV here or click to browse</p>
              <p className="text-slate-500 text-sm">COMPAS · UCI Adult · German Credit · or any CSV dataset</p>
            </>
          )}
        </div>
      )}

      {error && (
        <div className="mt-4 p-4 bg-red-900/40 border border-red-700 rounded-lg text-red-300 text-sm">
          {error}
        </div>
      )}

      {/* Column selector */}
      {columns.length > 0 && (
        <div className="mt-2">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-lg font-semibold text-white">Select Protected Attributes</h2>
              <p className="text-slate-400 text-sm">{rowCount.toLocaleString()} rows · {columns.length} columns</p>
            </div>
            <button
              onClick={() => { setColumns([]); setSelected(new Set()); setSessionId(null) }}
              className="text-xs text-slate-500 hover:text-slate-300 transition-colors"
            >
              ← Use different file
            </button>
          </div>

          <p className="text-slate-400 text-sm mb-1">
            Which columns should your model never be able to reconstruct?
          </p>
          <p className="text-slate-600 text-xs mb-3">
            Pick demographic attributes (gender, race, age group). ID columns, phone numbers, or
            unique identifiers will return no chains — they have no correlation with other features.
          </p>

          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 mb-6">
            {columns.map(col => (
              <button
                key={col.name}
                onClick={() => toggleCol(col.name)}
                className={`px-3 py-2 rounded-lg text-sm text-left border transition-all
                  ${selected.has(col.name)
                    ? 'bg-red-500/20 border-red-400 text-red-300 shadow-sm shadow-red-900'
                    : 'bg-slate-800 border-slate-600 text-slate-300 hover:border-slate-400 hover:bg-slate-700'}`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium truncate">{col.name}</span>
                  {selected.has(col.name) && <span className="ml-1 text-red-400">✓</span>}
                </div>
                <div className="text-xs text-slate-500 mt-0.5">{col.dtype} · {col.unique_count} unique</div>
              </button>
            ))}
          </div>

          <div className="flex items-center gap-3 mb-6">
            <label className="text-sm text-slate-400 whitespace-nowrap">Max chain depth:</label>
            {[2, 3, 4, 5, 6].map(d => (
              <button
                key={d}
                onClick={() => setMaxDepth(d)}
                className={`w-9 h-9 rounded-full text-sm font-bold transition-all
                  ${maxDepth === d
                    ? 'bg-red-500 text-white shadow-lg shadow-red-900'
                    : 'bg-slate-700 text-slate-400 hover:bg-slate-600'}`}
              >
                {d}
              </button>
            ))}
            <span className="text-xs text-slate-500 ml-1">
              {maxDepth === 2 ? 'Shallow - fast' : maxDepth <= 4 ? 'Recommended' : 'Deep - slower'}
            </span>
          </div>

          <button
            onClick={handleAudit}
            disabled={selected.size === 0 || auditing}
            className="w-full py-4 rounded-xl font-bold text-lg transition-all
              bg-red-500 hover:bg-red-600 hover:shadow-lg hover:shadow-red-900/40
              disabled:bg-slate-700 disabled:text-slate-500 text-white"
          >
            {auditing ? (
              <span className="flex items-center justify-center gap-3">
                <span className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Auditing dataset - tracing all paths…
              </span>
            ) : selected.size === 0
              ? 'Select at least one protected attribute'
              : `Run Audit → ${selected.size} protected attribute${selected.size !== 1 ? 's' : ''}`}
          </button>
        </div>
      )}

      {/* Evidence cards */}
      {!columns.length && !demoLoading && (
        <div className="mt-12 grid grid-cols-1 sm:grid-cols-3 gap-4">
          {[
            { co: 'Amazon Hiring AI', year: '2018', chain: 'University → Location → Demographics → Race', depth: '3-hop' },
            { co: 'COMPAS (ProPublica)', year: '2016', chain: 'Zip code → Arrests → Social network → Race', depth: '3-hop' },
            { co: 'Apple Card', year: '2019', chain: 'Marital status → Income → Credit → Gender', depth: '3-hop' },
          ].map(e => (
            <div key={e.co} className="bg-slate-800/50 border border-slate-700 rounded-xl p-4">
              <div className="text-xs text-red-400 font-bold mb-1">{e.depth}</div>
              <div className="text-white text-sm font-semibold mb-1">{e.co} ({e.year})</div>
              <div className="text-slate-500 text-xs font-mono">{e.chain}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
