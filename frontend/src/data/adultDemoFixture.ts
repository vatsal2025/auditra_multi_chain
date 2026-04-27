import { AuditResponse, UploadResponse } from '../services/api'

export const DEMO_UPLOAD: UploadResponse = {
  session_id: 'demo-adult-income',
  row_count: 48842,
  columns: [
    { name: 'age',            dtype: 'numeric',     unique_count: 73,  null_pct: 0 },
    { name: 'workclass',      dtype: 'categorical', unique_count: 8,   null_pct: 0.056 },
    { name: 'education',      dtype: 'categorical', unique_count: 16,  null_pct: 0 },
    { name: 'education_num',  dtype: 'numeric',     unique_count: 16,  null_pct: 0 },
    { name: 'marital_status', dtype: 'categorical', unique_count: 7,   null_pct: 0 },
    { name: 'occupation',     dtype: 'categorical', unique_count: 14,  null_pct: 0.056 },
    { name: 'relationship',   dtype: 'categorical', unique_count: 6,   null_pct: 0 },
    { name: 'race',           dtype: 'categorical', unique_count: 5,   null_pct: 0 },
    { name: 'sex',            dtype: 'categorical', unique_count: 2,   null_pct: 0 },
    { name: 'capital_gain',   dtype: 'numeric',     unique_count: 119, null_pct: 0 },
    { name: 'capital_loss',   dtype: 'numeric',     unique_count: 92,  null_pct: 0 },
    { name: 'hours_per_week', dtype: 'numeric',     unique_count: 94,  null_pct: 0 },
    { name: 'native_country', dtype: 'categorical', unique_count: 41,  null_pct: 0.017 },
    { name: 'income',         dtype: 'categorical', unique_count: 2,   null_pct: 0 },
  ],
}

export const DEMO_AUDIT: AuditResponse = {
  session_id: 'demo-adult-income',
  summary:
    'Found 10 relay chains across 2 protected attributes. 3 HIGH risk. ' +
    'Top chain: occupation → marital_status → relationship → sex (skill 0.51 — 51% above random baseline). ' +
    'SPD(sex)=−0.199, DI=0.364 — violates 80% rule (EU AI Act Art.10). ' +
    'Reweighing reduces |disc| from 0.199→0.109. Matches Amazon 2018 hiring AI pattern.',

  nodes: [
    { id: 'occupation',     label: 'occupation',     dtype: 'categorical', is_protected: false, risk_level: 'high' },
    { id: 'marital_status', label: 'marital_status', dtype: 'categorical', is_protected: false, risk_level: 'high' },
    { id: 'relationship',   label: 'relationship',   dtype: 'categorical', is_protected: false, risk_level: 'high' },
    { id: 'sex',            label: 'sex',            dtype: 'categorical', is_protected: true,  risk_level: 'high' },
    { id: 'education',      label: 'education',      dtype: 'categorical', is_protected: false, risk_level: 'medium' },
    { id: 'age',            label: 'age',            dtype: 'numeric',     is_protected: false, risk_level: 'medium' },
    { id: 'hours_per_week', label: 'hours_per_week', dtype: 'numeric',     is_protected: false, risk_level: 'medium' },
    { id: 'education_num',  label: 'education_num',  dtype: 'numeric',     is_protected: false, risk_level: 'medium' },
    { id: 'race',           label: 'race',           dtype: 'categorical', is_protected: true,  risk_level: 'low' },
    { id: 'income',         label: 'income',         dtype: 'categorical', is_protected: false, risk_level: 'none' },
    { id: 'workclass',      label: 'workclass',      dtype: 'categorical', is_protected: false, risk_level: 'low' },
    { id: 'capital_gain',   label: 'capital_gain',   dtype: 'numeric',     is_protected: false, risk_level: 'low' },
    { id: 'capital_loss',   label: 'capital_loss',   dtype: 'numeric',     is_protected: false, risk_level: 'low' },
    { id: 'native_country', label: 'native_country', dtype: 'categorical', is_protected: false, risk_level: 'low' },
  ],

  edges: [
    { source: 'relationship',   target: 'sex',            weight: 0.71 },
    { source: 'marital_status', target: 'sex',            weight: 0.62 },
    { source: 'occupation',     target: 'sex',            weight: 0.45 },
    { source: 'marital_status', target: 'relationship',   weight: 0.58 },
    { source: 'occupation',     target: 'marital_status', weight: 0.38 },
    { source: 'education',      target: 'occupation',     weight: 0.38 },
    { source: 'age',            target: 'marital_status', weight: 0.32 },
    { source: 'education',      target: 'income',         weight: 0.33 },
    { source: 'occupation',     target: 'income',         weight: 0.31 },
    { source: 'marital_status', target: 'income',         weight: 0.27 },
    { source: 'capital_gain',   target: 'income',         weight: 0.25 },
    { source: 'hours_per_week', target: 'occupation',     weight: 0.28 },
    { source: 'workclass',      target: 'occupation',     weight: 0.22 },
    { source: 'occupation',     target: 'race',           weight: 0.21 },
    { source: 'education',      target: 'race',           weight: 0.18 },
    { source: 'native_country', target: 'race',           weight: 0.19 },
    { source: 'age',            target: 'income',         weight: 0.19 },
    { source: 'education_num',  target: 'occupation',     weight: 0.35 },
  ],

  chains: [
    {
      id: 'c001', protected_attribute: 'sex', risk_score: 0.5122, risk_label: 'HIGH',
      path: ['occupation', 'marital_status', 'relationship', 'sex'],
      hops: [
        { source: 'occupation',     target: 'marital_status', weight: 0.38 },
        { source: 'marital_status', target: 'relationship',   weight: 0.58 },
        { source: 'relationship',   target: 'sex',            weight: 0.71 },
      ],
      weakest_link: 'occupation',
      explanation:
        'occupation → marital_status → relationship forms a 3-hop relay reconstructing sex with 51.2% skill above random baseline — the exact pattern behind Amazon\'s 2018 hiring AI scandal. Removing \'occupation\' breaks the chain.',
    },
    {
      id: 'c002', protected_attribute: 'sex', risk_score: 0.4234, risk_label: 'HIGH',
      path: ['education', 'marital_status', 'relationship', 'sex'],
      hops: [
        { source: 'education',      target: 'marital_status', weight: 0.29 },
        { source: 'marital_status', target: 'relationship',   weight: 0.58 },
        { source: 'relationship',   target: 'sex',            weight: 0.71 },
      ],
      weakest_link: 'education',
      explanation:
        'Education level → marital status → relationship type reconstructs sex (42.3% skill). EU AI Act Article 10 prohibits indirect discrimination via educational proxies.',
    },
    {
      id: 'c003', protected_attribute: 'sex', risk_score: 0.3856, risk_label: 'HIGH',
      path: ['age', 'marital_status', 'relationship', 'sex'],
      hops: [
        { source: 'age',            target: 'marital_status', weight: 0.32 },
        { source: 'marital_status', target: 'relationship',   weight: 0.58 },
        { source: 'relationship',   target: 'sex',            weight: 0.71 },
      ],
      weakest_link: 'age',
      explanation: null,
    },
    {
      id: 'c004', protected_attribute: 'sex', risk_score: 0.2734, risk_label: 'MEDIUM',
      path: ['hours_per_week', 'occupation', 'marital_status', 'sex'],
      hops: [
        { source: 'hours_per_week', target: 'occupation',     weight: 0.28 },
        { source: 'occupation',     target: 'marital_status', weight: 0.38 },
        { source: 'marital_status', target: 'sex',            weight: 0.62 },
      ],
      weakest_link: 'hours_per_week',
      explanation: null,
    },
    {
      id: 'c005', protected_attribute: 'sex', risk_score: 0.2341, risk_label: 'MEDIUM',
      path: ['workclass', 'occupation', 'relationship', 'sex'],
      hops: [
        { source: 'workclass',  target: 'occupation',   weight: 0.22 },
        { source: 'occupation', target: 'relationship', weight: 0.35 },
        { source: 'relationship', target: 'sex',        weight: 0.71 },
      ],
      weakest_link: 'workclass',
      explanation: null,
    },
    {
      id: 'c006', protected_attribute: 'sex', risk_score: 0.2198, risk_label: 'MEDIUM',
      path: ['education_num', 'marital_status', 'relationship', 'sex'],
      hops: [
        { source: 'education_num',  target: 'marital_status', weight: 0.26 },
        { source: 'marital_status', target: 'relationship',   weight: 0.58 },
        { source: 'relationship',   target: 'sex',            weight: 0.71 },
      ],
      weakest_link: 'education_num',
      explanation: null,
    },
    {
      id: 'c007', protected_attribute: 'sex', risk_score: 0.1823, risk_label: 'MEDIUM',
      path: ['native_country', 'marital_status', 'sex'],
      hops: [
        { source: 'native_country', target: 'marital_status', weight: 0.21 },
        { source: 'marital_status', target: 'sex',            weight: 0.62 },
      ],
      weakest_link: 'native_country',
      explanation: null,
    },
    {
      id: 'c008', protected_attribute: 'sex', risk_score: 0.1456, risk_label: 'MEDIUM',
      path: ['capital_gain', 'occupation', 'sex'],
      hops: [
        { source: 'capital_gain', target: 'occupation', weight: 0.19 },
        { source: 'occupation',   target: 'sex',        weight: 0.45 },
      ],
      weakest_link: 'capital_gain',
      explanation: null,
    },
    {
      id: 'c009', protected_attribute: 'race', risk_score: 0.1234, risk_label: 'LOW',
      path: ['occupation', 'marital_status', 'race'],
      hops: [
        { source: 'occupation',     target: 'marital_status', weight: 0.38 },
        { source: 'marital_status', target: 'race',           weight: 0.24 },
      ],
      weakest_link: 'occupation',
      explanation: null,
    },
    {
      id: 'c010', protected_attribute: 'race', risk_score: 0.0987, risk_label: 'LOW',
      path: ['education', 'occupation', 'race'],
      hops: [
        { source: 'education', target: 'occupation', weight: 0.38 },
        { source: 'occupation', target: 'race',      weight: 0.21 },
      ],
      weakest_link: 'education',
      explanation: null,
    },
  ],

  fairness_metrics: [
    {
      protected_attribute: 'sex', outcome_column: 'income',
      privileged_group: 'Male', positive_outcome: '>50K',
      statistical_parity_diff: -0.1989, disparate_impact_ratio: 0.3635,
      equal_opportunity_diff: -0.051, average_odds_diff: -0.089,
      predictive_parity_diff: -0.112, model_accuracy_overall: 0.847,
      group_metrics: {
        Male:   { group_value: 'Male',   size: 5421, base_rate: 0.311, prediction_rate: 0.318, tpr: 0.789, fpr: 0.152, precision: 0.728, accuracy: 0.862 },
        Female: { group_value: 'Female', size: 2579, base_rate: 0.109, prediction_rate: 0.115, tpr: 0.738, fpr: 0.087, precision: 0.625, accuracy: 0.902 },
      },
    },
    {
      protected_attribute: 'race', outcome_column: 'income',
      privileged_group: 'White', positive_outcome: '>50K',
      statistical_parity_diff: -0.162, disparate_impact_ratio: 0.6038,
      equal_opportunity_diff: -0.082, average_odds_diff: -0.045,
      predictive_parity_diff: -0.078, model_accuracy_overall: 0.851,
      group_metrics: {
        White:     { group_value: 'White',     size: 6831, base_rate: 0.261, prediction_rate: 0.268, tpr: 0.784, fpr: 0.132, precision: 0.712, accuracy: 0.851 },
        'Non-White': { group_value: 'Non-White', size: 1169, base_rate: 0.126, prediction_rate: 0.113, tpr: 0.672, fpr: 0.089, precision: 0.618, accuracy: 0.891 },
      },
    },
  ],

  mitigated_fairness_metrics: [
    {
      protected_attribute: 'sex', outcome_column: 'income',
      privileged_group: 'Male', positive_outcome: '>50K',
      statistical_parity_diff: -0.109, disparate_impact_ratio: 0.527,
      equal_opportunity_diff: 0.117, average_odds_diff: 0.042,
      predictive_parity_diff: -0.031, model_accuracy_overall: 0.831,
      group_metrics: {
        Male:   { group_value: 'Male',   size: 5421, base_rate: 0.311, prediction_rate: 0.298, tpr: 0.801, fpr: 0.141, precision: 0.714, accuracy: 0.849 },
        Female: { group_value: 'Female', size: 2579, base_rate: 0.109, prediction_rate: 0.189, tpr: 0.918, fpr: 0.099, precision: 0.589, accuracy: 0.878 },
      },
    },
    {
      protected_attribute: 'race', outcome_column: 'income',
      privileged_group: 'White', positive_outcome: '>50K',
      statistical_parity_diff: -0.043, disparate_impact_ratio: 0.843,
      equal_opportunity_diff: 0.038, average_odds_diff: 0.012,
      predictive_parity_diff: -0.019, model_accuracy_overall: 0.841,
      group_metrics: {
        White:     { group_value: 'White',     size: 6831, base_rate: 0.261, prediction_rate: 0.254, tpr: 0.791, fpr: 0.128, precision: 0.708, accuracy: 0.847 },
        'Non-White': { group_value: 'Non-White', size: 1169, base_rate: 0.126, prediction_rate: 0.211, tpr: 0.829, fpr: 0.102, precision: 0.634, accuracy: 0.871 },
      },
    },
  ],
}
