export function SkeletonBox({ className = '' }: { className?: string }) {
  return (
    <div className={`animate-pulse bg-slate-700/50 rounded-lg ${className}`} />
  )
}

export function GraphSkeleton() {
  return (
    <div className="w-full rounded-xl bg-slate-900 border border-slate-700 flex items-center justify-center"
      style={{ height: 500 }}>
      <div className="text-center">
        <div className="w-12 h-12 border-2 border-red-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
        <p className="text-slate-400 text-sm">Building feature graph…</p>
        <p className="text-slate-500 text-xs mt-1">Computing correlations across all column pairs</p>
      </div>
    </div>
  )
}

export function ChainSkeleton() {
  return (
    <div className="space-y-2">
      {[1, 2, 3].map(i => (
        <div key={i} className="bg-slate-800 border border-slate-700 rounded-lg p-3">
          <SkeletonBox className="h-3 w-24 mb-2" />
          <SkeletonBox className="h-3 w-48" />
          <SkeletonBox className="h-2 w-32 mt-2" />
        </div>
      ))}
    </div>
  )
}
