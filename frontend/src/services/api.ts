import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

export interface ColumnInfo {
  name: string
  dtype: string
  unique_count: number
  null_pct: number
}

export interface UploadResponse {
  session_id: string
  columns: ColumnInfo[]
  row_count: number
}

export interface ChainHop {
  source: string
  target: string
  weight: number
}

export interface Chain {
  id: string
  path: string[]
  hops: ChainHop[]
  risk_score: number
  risk_label: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
  protected_attribute: string
  explanation: string | null
  weakest_link: string | null
}

export interface GraphNode {
  id: string
  label: string
  dtype: string
  is_protected: boolean
  risk_level: string
}

export interface GraphEdge {
  source: string
  target: string
  weight: number
}

export interface GroupMetrics {
  group_value: string
  size: number
  base_rate: number
  prediction_rate: number
  tpr: number
  fpr: number
  precision: number
  accuracy: number
}

export interface FairnessMetrics {
  protected_attribute: string
  outcome_column: string
  privileged_group: string
  positive_outcome: string
  statistical_parity_diff: number
  disparate_impact_ratio: number
  equal_opportunity_diff: number
  average_odds_diff: number
  predictive_parity_diff: number
  model_accuracy_overall: number
  group_metrics: Record<string, GroupMetrics>
}

export interface AuditResponse {
  session_id: string
  nodes: GraphNode[]
  edges: GraphEdge[]
  chains: Chain[]
  summary: string
  fairness_metrics: FairnessMetrics[]
  mitigated_fairness_metrics: FairnessMetrics[]
}

export interface ShapEntry {
  feature: string
  before: number
  after: number
}

export interface FixResponse {
  session_id: string
  chain_id: string
  removed_feature: string
  shap_values: ShapEntry[]
  success: boolean
  message: string
}

export const uploadDataset = (file: File) => {
  const form = new FormData()
  form.append('file', file)
  return api.post<UploadResponse>('/upload', form)
}

export const runAudit = (
  session_id: string,
  protected_attributes: string[],
  max_depth = 4,
  threshold = 0.15,
  outcome_column?: string,
) =>
  api.post<AuditResponse>('/audit', { session_id, protected_attributes, max_depth, threshold, outcome_column: outcome_column || null })

export const applyFix = (session_id: string, chain_id: string) =>
  api.post<FixResponse>('/fix', { session_id, chain_id })

export const sendChat = (session_id: string, message: string) =>
  api.post<{ reply: string }>('/chat', { session_id, message })

export const createReport = (session_id: string) =>
  api.post<{ download_url: string }>('/report', { session_id })
