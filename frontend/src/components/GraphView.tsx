import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import { AuditResponse, Chain, GraphNode } from '../services/api'

interface Props {
  audit: AuditResponse
  selectedChain: Chain | null
  onNodeClick: (nodeId: string) => void
}

interface SimNode extends GraphNode, d3.SimulationNodeDatum {
  fx?: number | null
  fy?: number | null
}

const RISK_COLORS: Record<string, string> = {
  critical: '#ef4444',
  high: '#f97316',
  medium: '#eab308',
  low: '#22c55e',
  none: '#475569',
}

export default function GraphView({ audit, selectedChain, onNodeClick }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement>(null)
  const [isFullscreen, setIsFullscreen] = useState(false)

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      containerRef.current?.requestFullscreen()
    } else {
      document.exitFullscreen()
    }
  }

  useEffect(() => {
    const handler = () => setIsFullscreen(!!document.fullscreenElement)
    document.addEventListener('fullscreenchange', handler)
    return () => document.removeEventListener('fullscreenchange', handler)
  }, [])

  useEffect(() => {
    const el = svgRef.current
    const container = containerRef.current
    if (!el || !container) return

    const width = container.clientWidth || 900
    const height = container.clientHeight || 560

    d3.select(el).selectAll('*').remove()

    const svg = d3.select(el)
      .attr('width', width)
      .attr('height', height)

    const highlightedNodes = new Set(selectedChain?.path ?? [])
    const highlightedEdges = new Set(
      selectedChain?.hops.map(h => `${h.source}__${h.target}`) ?? []
    )

    // Zoomable group
    const g = svg.append('g')

    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.15, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform)
      })

    svg.call(zoom)

    // Arrow markers (inside zoomable group so they scale)
    g.append('defs').selectAll('marker')
      .data(['default', 'highlighted'])
      .join('marker')
      .attr('id', d => `arrow-${d}`)
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 42)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', d => d === 'highlighted' ? '#ef4444' : '#e2e8f0')

    const nodesCopy: SimNode[] = audit.nodes.map(n => ({ ...n }))
    const edgesCopy = audit.edges.map(e => ({ ...e }))

    const simulation = d3.forceSimulation<SimNode>(nodesCopy)
      .force('link', d3.forceLink(edgesCopy)
        .id((d: any) => d.id)
        .distance(170))
      .force('charge', d3.forceManyBody().strength(-600))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide(68))

    const link = g.append('g').selectAll('line')
      .data(edgesCopy)
      .join('line')
      .attr('stroke', d => {
        const key = `${(d.source as any).id ?? d.source}__${(d.target as any).id ?? d.target}`
        return highlightedEdges.has(key) ? '#ef4444' : '#e2e8f0'
      })
      .attr('stroke-width', d => {
        const key = `${(d.source as any).id ?? d.source}__${(d.target as any).id ?? d.target}`
        return highlightedEdges.has(key) ? 2.5 : 1.2
      })
      .attr('stroke-opacity', d => {
        const key = `${(d.source as any).id ?? d.source}__${(d.target as any).id ?? d.target}`
        return highlightedEdges.has(key) ? 1 : 0.55
      })
      .attr('filter', 'url(#glow-line)')
      .attr('marker-end', d => {
        const key = `${(d.source as any).id ?? d.source}__${(d.target as any).id ?? d.target}`
        return `url(#arrow-${highlightedEdges.has(key) ? 'highlighted' : 'default'})`
      })

    const node = g.append('g').selectAll<SVGGElement, SimNode>('g')
      .data(nodesCopy)
      .join('g')
      .attr('cursor', 'grab')
      .on('click', (event, d) => {
        event.stopPropagation()
        onNodeClick(d.id)
      })
      .call(
        d3.drag<SVGGElement, SimNode>()
          .on('start', (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart()
            d.fx = d.x; d.fy = d.y
          })
          .on('drag', (event, d) => {
            d.fx = event.x; d.fy = event.y
          })
          .on('end', (event, d) => {
            if (!event.active) simulation.alphaTarget(0)
            d.fx = event.x; d.fy = event.y
          })
      )

    // Double-click to unpin
    node.on('dblclick', (event, d) => {
      event.stopPropagation()
      d.fx = null; d.fy = null
      simulation.alphaTarget(0.2).restart()
    })

    // Glow filter for highlighted nodes
    const defs = g.select('defs')
    const lineFilter = defs.append('filter').attr('id', 'glow-line')
    lineFilter.append('feGaussianBlur').attr('stdDeviation', 2).attr('result', 'blur')
    const lMerge = lineFilter.append('feMerge')
    lMerge.append('feMergeNode').attr('in', 'blur')
    lMerge.append('feMergeNode').attr('in', 'SourceGraphic')
    const filter = defs.append('filter').attr('id', 'glow')
    filter.append('feGaussianBlur').attr('stdDeviation', 3.5).attr('result', 'blur')
    const feMerge = filter.append('feMerge')
    feMerge.append('feMergeNode').attr('in', 'blur')
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic')

    node.append('circle')
      .attr('r', d => d.is_protected ? 38 : 30)
      .attr('fill', d => RISK_COLORS[d.risk_level] ?? '#475569')
      .attr('fill-opacity', d => highlightedNodes.has(d.id) ? 1 : 0.55)
      .attr('stroke', d => highlightedNodes.has(d.id) ? '#fff' : 'rgba(255,255,255,0.15)')
      .attr('stroke-width', d => highlightedNodes.has(d.id) ? 3 : 1.5)
      .attr('filter', d => highlightedNodes.has(d.id) ? 'url(#glow)' : '')

    node.append('text')
      .text(d => d.label.length > 13 ? d.label.slice(0, 11) + '…' : d.label)
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .attr('font-size', '11px')
      .attr('font-weight', d => highlightedNodes.has(d.id) ? '700' : '500')
      .attr('fill', '#f1f5f9')
      .attr('pointer-events', 'none')

    simulation.on('tick', () => {
      link
        .attr('x1', d => (d.source as any).x)
        .attr('y1', d => (d.source as any).y)
        .attr('x2', d => (d.target as any).x)
        .attr('y2', d => (d.target as any).y)
      node.attr('transform', d => `translate(${d.x ?? 0},${d.y ?? 0})`)
    })

    // Fit view hint
    svg.append('text')
      .attr('x', 10).attr('y', height - 10)
      .attr('fill', '#475569').attr('font-size', '10px')
      .text('Drag nodes · Scroll to zoom · Drag canvas to pan · Dbl-click node to unpin')

    return () => { simulation.stop() }
  }, [audit, selectedChain, onNodeClick, isFullscreen])

  return (
    <div
      ref={containerRef}
      className="relative w-full rounded-xl bg-slate-900 border border-slate-700 overflow-hidden"
      style={{ height: isFullscreen ? '100vh' : '62vh', minHeight: isFullscreen ? '100vh' : 480 }}
    >
      <button
        onClick={toggleFullscreen}
        className="absolute top-3 right-3 z-10 bg-slate-800/90 hover:bg-slate-700 border border-slate-600 text-slate-300 hover:text-white rounded-lg px-3 py-1.5 text-xs font-medium transition-colors backdrop-blur-sm"
      >
        {isFullscreen ? '✕ Exit Fullscreen' : '⛶ Fullscreen'}
      </button>
      <svg ref={svgRef} className="w-full h-full" />
    </div>
  )
}
