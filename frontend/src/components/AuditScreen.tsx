import { useState } from 'react'
import {
  AuditResponse,
  Chain,
  createReport,
  FixResponse,
  UploadResponse,
} from '../services/api'
import GraphView from './GraphView'
import ChainPanel from './ChainPanel'
import ChatBox from './ChatBox'
import ShapChart from './ShapChart'
import FairnessPanel from './FairnessPanel'
import { GraphSkeleton } from './Skeleton'
import ToastContainer, { useToast } from './Toast'

interface Props {
  uploadData: UploadResponse
  initialAuditData: AuditResponse | null
  onAuditComplete: (data: AuditResponse) => void
}

function CleanDatasetBanner({ summary }: { summary: string }) {
  const reasons = [
    'Dataset may not contain proxies for the protected attribute.',
    'Protected attribute may be truly independent of other columns.',
    'Data could be an ID/contact table (phone numbers, emails) with no demographic signal.',
    'Try selecting a different protected attribute, or lower the threshold.',
  ]
  return (
    <div className="mt-4 p-6 bg-slate-800/60 border border-slate-600 rounded-xl">
      <div className="flex items-start gap-4">
        <span className="text-4xl shrink-0">🎉</span>
        <div>
          <h3 className="text-lg font-bold text-white mb-1">No bias chains found</h3>
          <p className="text-slate-400 text-sm mb-4">{summary || 'No multi-hop discrimination paths detected at the current threshold and depth.'}</p>
          <p className="text-slate-500 text-xs font-semibold uppercase tracking-wider mb-2">Possible reasons</p>
          <ul className="space-y-1">
            {reasons.map(r => (
              <li key={r} className="text-slate-400 text-xs flex items-start gap-2">
                <span className="text-slate-600 mt-0.5">•</span>{r}
              </li>
            ))}
          </ul>
          <p className="mt-4 text-xs text-slate-500">
            The graph above still shows all feature relationships — hover nodes to explore correlations.
          </p>
        </div>
      </div>
    </div>
  )
}

export default function AuditScreen({ uploadData, initialAuditData, onAuditComplete }: Props) {
  const [audit, setAudit] = useState<AuditResponse | null>(initialAuditData)
  const [selectedChain, setSelectedChain] = useState<Chain | null>(null)
  const [lastFix, setLastFix] = useState<FixResponse | null>(null)
  const [activeTab, setActiveTab] = useState<'chains' | 'fairness' | 'chat'>('chains')
  const [reportLoading, setReportLoading] = useState(false)
  const [reportUrl, setReportUrl] = useState<string | null>(null)
  const { toasts, dismiss, toast } = useToast()

  const handleFixApplied = (fix: FixResponse) => {
    setLastFix(fix)
    if (!audit) return
    const removed = fix.removed_feature
    const remainingChains = audit.chains.filter(c =>
      c.id !== fix.chain_id &&
      (removed === '' || c.weakest_link !== removed)
    )
    const remainingNodes = removed
      ? audit.nodes.filter(n => n.id !== removed)
      : audit.nodes
    const remainingEdges = removed
      ? audit.edges.filter(e => e.source !== removed && e.target !== removed)
      : audit.edges
    const updated = { ...audit, chains: remainingChains, nodes: remainingNodes, edges: remainingEdges }
    setAudit(updated)
    onAuditComplete(updated)
    setSelectedChain(null)
    toast?.(`Removed '${removed}' - chain broken`, 'success')
  }

  const handleReport = async () => {
    setReportLoading(true)
    try {
      const res = await createReport(uploadData.session_id)
      setReportUrl(res.data.download_url)
      toast?.('Report ready - click Download', 'success')
    } catch {
      toast?.('Report generation failed', 'error')
    } finally {
      setReportLoading(false)
    }
  }

  if (!audit) {
    return (
      <div className="max-w-7xl mx-auto">
        <GraphSkeleton />
      </div>
    )
  }

  const critical = audit.chains.filter(c => c.risk_label === 'CRITICAL').length
  const high = audit.chains.filter(c => c.risk_label === 'HIGH').length
  const medium = audit.chains.filter(c => c.risk_label === 'MEDIUM').length
  const compliant = critical === 0 && high === 0
  const hasFairness = (audit.fairness_metrics?.length ?? 0) > 0

  return (
    <>
      <ToastContainer toasts={toasts} onDismiss={dismiss} />

      <div className="max-w-[1600px] mx-auto space-y-5">
        {/* Summary bar */}
        <div className="flex items-center gap-4 flex-wrap">
          <div className="flex items-center gap-3">
            <div>
              <span className="text-3xl font-bold text-white">{audit.chains.length}</span>
              <span className="text-slate-400 ml-2 text-sm">chains</span>
            </div>
            {critical > 0 && (
              <span className="px-3 py-1 bg-red-900/60 border border-red-700 rounded-full text-red-300 text-xs font-bold">
                {critical} CRITICAL
              </span>
            )}
            {high > 0 && (
              <span className="px-3 py-1 bg-orange-900/60 border border-orange-700 rounded-full text-orange-300 text-xs font-bold">
                {high} HIGH
              </span>
            )}
            {medium > 0 && (
              <span className="px-3 py-1 bg-yellow-900/60 border border-yellow-700 rounded-full text-yellow-300 text-xs font-bold">
                {medium} MEDIUM
              </span>
            )}
          </div>

          <div className={`ml-2 flex items-center gap-2 px-3 py-1 rounded-full border text-xs font-semibold
            ${compliant
              ? 'bg-green-900/40 border-green-700 text-green-300'
              : 'bg-red-900/40 border-red-700 text-red-300'}`}>
            {compliant ? '✅ EU AI Act Compliant' : '❌ Non-Compliant — fixes required'}
          </div>

          <p className="text-slate-400 text-sm hidden lg:block ml-2 max-w-lg truncate">{audit.summary}</p>

          <div className="ml-auto flex gap-2">
            {reportUrl ? (
              <a
                href={reportUrl}
                target="_blank"
                rel="noreferrer"
                className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white text-sm font-semibold rounded-lg transition-colors"
              >
                ↓ Download Report
              </a>
            ) : (
              <button
                onClick={handleReport}
                disabled={reportLoading}
                className="px-4 py-2 bg-slate-700 hover:bg-slate-600 disabled:opacity-50
                  text-white text-sm font-semibold rounded-lg transition-colors"
              >
                {reportLoading ? (
                  <span className="flex items-center gap-2">
                    <span className="w-3 h-3 border border-white border-t-transparent rounded-full animate-spin" />
                    Generating…
                  </span>
                ) : 'Generate Report'}
              </button>
            )}
          </div>
        </div>

        {/* Full-width graph */}
        <div>
          <GraphView
            audit={audit}
            selectedChain={selectedChain}
            onNodeClick={(nodeId) => {
              const chain = audit.chains.find(c => c.path.includes(nodeId))
              if (chain) {
                setSelectedChain(chain)
                setActiveTab('chains')
                toast?.(`Selected: ${chain.path.join(' → ')}`, 'info')
              }
            }}
          />

          {/* Legend */}
          <div className="flex flex-wrap gap-4 px-2 mt-2">
            {[
              { label: 'CRITICAL', color: 'bg-red-500' },
              { label: 'HIGH', color: 'bg-orange-500' },
              { label: 'MEDIUM', color: 'bg-yellow-500' },
              { label: 'LOW', color: 'bg-green-500' },
              { label: 'No chain', color: 'bg-slate-500' },
            ].map(l => (
              <div key={l.label} className="flex items-center gap-1.5 text-xs text-slate-400">
                <span className={`w-2.5 h-2.5 rounded-full ${l.color}`} />
                {l.label}
              </div>
            ))}
            <span className="text-xs text-slate-500 ml-auto">Click node to select chain</span>
          </div>
        </div>

        {lastFix && <ShapChart entries={lastFix.shap_values} />}

        {/* 3-column panel row */}
        <div className={`grid gap-5 ${hasFairness ? 'grid-cols-1 xl:grid-cols-3' : 'grid-cols-1 xl:grid-cols-2'}`}>
          {/* Chains panel */}
          <div
            className="bg-slate-900 border border-slate-700 rounded-xl p-4 flex flex-col"
            style={{ height: 560 }}
          >
            <div className="flex items-center justify-between mb-3 shrink-0">
              <h3 className="text-white font-semibold text-sm">
                Relay Chains
                <span className="ml-2 text-slate-500 font-normal">({audit.chains.length})</span>
              </h3>
            </div>
            <div className="flex-1 overflow-hidden">
              <ChainPanel
                chains={audit.chains}
                selectedChain={selectedChain}
                sessionId={uploadData.session_id}
                onSelectChain={setSelectedChain}
                onFixApplied={handleFixApplied}
              />
            </div>
          </div>

          {/* Fairness metrics panel */}
          {hasFairness && (
            <div
              className="bg-slate-900 border border-slate-700 rounded-xl p-4 flex flex-col"
              style={{ height: 560 }}
            >
              <div className="flex items-center justify-between mb-3 shrink-0">
                <h3 className="text-white font-semibold text-sm flex items-center gap-2">
                  Fairness Metrics
                  <span className="px-1.5 py-0.5 bg-blue-900/50 border border-blue-700/50 text-blue-300 text-xs rounded font-mono">
                    SPD · DI · EOD · AOD
                  </span>
                </h3>
              </div>
              <div className="flex-1 overflow-y-auto min-h-0">
                <FairnessPanel
                  metrics={audit.fairness_metrics}
                  mitigatedMetrics={audit.mitigated_fairness_metrics ?? []}
                />
              </div>
            </div>
          )}

          {/* Chat panel */}
          <div
            className="bg-slate-900 border border-slate-700 rounded-xl p-4 flex flex-col"
            style={{ height: 560 }}
          >
            <div className="flex gap-1 mb-3 shrink-0">
              <button
                onClick={() => setActiveTab('chat')}
                className="flex-1 py-2 rounded-lg text-sm font-bold transition-all
                  bg-red-600 hover:bg-red-500 text-white shadow-lg shadow-red-700/60
                  ring-1 ring-red-400/50 hover:ring-red-400 hover:shadow-red-500/70"
              >
                💬 Ask Gemini
              </button>
            </div>
<div className="flex-1 min-h-0 overflow-hidden">
              <ChatBox sessionId={uploadData.session_id} />
            </div>
          </div>
        </div>

        {/* Empty / clean state */}
        {audit.chains.length === 0 && <CleanDatasetBanner summary={audit.summary} />}

        {/* All-LOW advisory */}
        {audit.chains.length > 0 && high === 0 && critical === 0 && medium === 0 && (
          <div className="p-4 bg-green-950/40 border border-green-700/40 rounded-xl flex items-start gap-3">
            <span className="text-2xl shrink-0">✅</span>
            <div>
              <p className="text-green-300 font-semibold text-sm">No high-risk patterns detected</p>
              <p className="text-slate-400 text-xs mt-1">
                All {audit.chains.length} chains scored LOW — no feature combination reconstructs the
                protected attribute significantly above random chance. This dataset looks clean for
                the selected attributes. Try a lower threshold or deeper depth to be thorough.
              </p>
            </div>
          </div>
        )}
      </div>
    </>
  )
}
