import { useEffect, useRef } from 'react'
import * as d3 from 'd3'
import { AuditResponse, Chain } from '../services/api'

interface Props {
  audit: AuditResponse
  selectedChain: Chain | null
  onNodeClick: (nodeId: string) => void
}

const RISK_COLORS: Record<string, string> = {
  critical: '#ef4444',
  high: '#f97316',
  medium: '#eab308',
  low: '#22c55e',
  none: '#475569',
}

export default function GraphView({ audit, selectedChain, onNodeClick }: Props) {
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    if (!svgRef.current) return

    const width = svgRef.current.clientWidth || 800
    const height = 500

    d3.select(svgRef.current).selectAll('*').remove()

    const svg = d3.select(svgRef.current)
      .attr('viewBox', `0 0 ${width} ${height}`)

    // Highlight nodes/edges in selected chain
    const highlightedNodes = new Set(selectedChain?.path ?? [])
    const highlightedEdges = new Set(
      selectedChain?.hops.map(h => `${h.source}__${h.target}`) ?? []
    )

    const simulation = d3.forceSimulation(audit.nodes as any)
      .force('link', d3.forceLink(audit.edges)
        .id((d: any) => d.id)
        .distance(120))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide(40))

    // Arrow markers
    svg.append('defs').selectAll('marker')
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
      .attr('fill', d => d === 'highlighted' ? '#ef4444' : '#475569')

    const link = svg.append('g').selectAll('line')
      .data(audit.edges)
      .join('line')
      .attr('stroke', d => {
        const key = `${(d.source as any).id ?? d.source}__${(d.target as any).id ?? d.target}`
        return highlightedEdges.has(key) ? '#ef4444' : '#334155'
      })
      .attr('stroke-width', d => {
        const key = `${(d.source as any).id ?? d.source}__${(d.target as any).id ?? d.target}`
        return highlightedEdges.has(key) ? 2.5 : 1
      })
      .attr('stroke-opacity', 0.8)
      .attr('marker-end', d => {
        const key = `${(d.source as any).id ?? d.source}__${(d.target as any).id ?? d.target}`
        return `url(#arrow-${highlightedEdges.has(key) ? 'highlighted' : 'default'})`
      })

    const node = svg.append('g').selectAll('g')
      .data(audit.nodes)
      .join('g')
      .attr('cursor', 'pointer')
      .on('click', (_, d) => onNodeClick(d.id))
      .call(
        d3.drag<SVGGElement, any>()
          .on('start', (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart()
            d.fx = d.x; d.fy = d.y
          })
          .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y })
          .on('end', (event, d) => {
            if (!event.active) simulation.alphaTarget(0)
            d.fx = null; d.fy = null
          }) as any
      )

    node.append('circle')
      .attr('r', d => d.is_protected ? 22 : 16)
      .attr('fill', d => RISK_COLORS[d.risk_level] ?? '#475569')
      .attr('fill-opacity', d => highlightedNodes.has(d.id) ? 1 : 0.5)
      .attr('stroke', d => highlightedNodes.has(d.id) ? '#fff' : 'transparent')
      .attr('stroke-width', 2)

    node.append('text')
      .text(d => d.label.length > 14 ? d.label.slice(0, 12) + '…' : d.label)
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .attr('font-size', '10px')
      .attr('fill', '#f1f5f9')
      .attr('pointer-events', 'none')

    simulation.on('tick', () => {
      link
        .attr('x1', d => (d.source as any).x)
        .attr('y1', d => (d.source as any).y)
        .attr('x2', d => (d.target as any).x)
        .attr('y2', d => (d.target as any).y)

      node.attr('transform', d => `translate(${(d as any).x},${(d as any).y})`)
    })

    return () => { simulation.stop() }
  }, [audit, selectedChain, onNodeClick])

  return (
    <svg
      ref={svgRef}
      className="w-full rounded-xl bg-slate-900 border border-slate-700"
      style={{ height: 500 }}
    />
  )
}
