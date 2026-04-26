import { useEffect, useRef } from 'react'
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
      .attr('refX', 28)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', d => d === 'highlighted' ? '#ef4444' : '#334155')

    const nodesCopy: SimNode[] = audit.nodes.map(n => ({ ...n }))
    const edgesCopy = audit.edges.map(e => ({ ...e }))

    const simulation = d3.forceSimulation<SimNode>(nodesCopy)
      .force('link', d3.forceLink(edgesCopy)
        .id((d: any) => d.id)
        .distance(130))
      .force('charge', d3.forceManyBody().strength(-380))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide(44))

    const link = g.append('g').selectAll('line')
      .data(edgesCopy)
      .join('line')
      .attr('stroke', d => {
        const key = `${(d.source as any).id ?? d.source}__${(d.target as any).id ?? d.target}`
        return highlightedEdges.has(key) ? '#ef4444' : '#334155'
      })
      .attr('stroke-width', d => {
        const key = `${(d.source as any).id ?? d.source}__${(d.target as any).id ?? d.target}`
        return highlightedEdges.has(key) ? 2.5 : 1.2
      })
      .attr('stroke-opacity', 0.9)
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
    const filter = defs.append('filter').attr('id', 'glow')
    filter.append('feGaussianBlur').attr('stdDeviation', 3.5).attr('result', 'blur')
    const feMerge = filter.append('feMerge')
    feMerge.append('feMergeNode').attr('in', 'blur')
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic')

    node.append('circle')
      .attr('r', d => d.is_protected ? 24 : 17)
      .attr('fill', d => RISK_COLORS[d.risk_level] ?? '#475569')
      .attr('fill-opacity', d => highlightedNodes.has(d.id) ? 1 : 0.45)
      .attr('stroke', d => highlightedNodes.has(d.id) ? '#fff' : 'transparent')
      .attr('stroke-width', 2.5)
      .attr('filter', d => highlightedNodes.has(d.id) ? 'url(#glow)' : '')

    node.append('text')
      .text(d => d.label.length > 14 ? d.label.slice(0, 12) + '…' : d.label)
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .attr('font-size', '10px')
      .attr('font-weight', d => highlightedNodes.has(d.id) ? '700' : '400')
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
  }, [audit, selectedChain, onNodeClick])

  return (
    <div
      ref={containerRef}
      className="w-full rounded-xl bg-slate-900 border border-slate-700 overflow-hidden"
      style={{ height: '62vh', minHeight: 480 }}
    >
      <svg ref={svgRef} className="w-full h-full" />
    </div>
  )
}
