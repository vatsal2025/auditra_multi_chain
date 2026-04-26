from pydantic import BaseModel
from typing import Dict, List, Optional, Any


class ColumnInfo(BaseModel):
    name: str
    dtype: str          # "numeric" | "categorical"
    unique_count: int
    null_pct: float


class UploadResponse(BaseModel):
    session_id: str
    columns: List[ColumnInfo]
    row_count: int


class AuditRequest(BaseModel):
    session_id: str
    protected_attributes: List[str]
    max_depth: int = 4
    threshold: float = 0.15
    # Optional: for fairness metrics computation
    outcome_column: Optional[str] = None
    privileged_groups: Optional[Dict[str, str]] = None
    positive_outcome: Optional[str] = None
    # Fast mode: skip conjunctive/intersectional/calibration (used by demo cold path)
    fast_mode: bool = False


class ChainHop(BaseModel):
    source: str
    target: str
    weight: float       # correlation / predictive strength 0-1


class Chain(BaseModel):
    id: str
    path: List[str]
    hops: List[ChainHop]
    risk_score: float   # 0-1, skill above baseline
    risk_label: str     # LOW | MEDIUM | HIGH | CRITICAL
    protected_attribute: str
    explanation: Optional[str] = None
    weakest_link: Optional[str] = None


class GraphNode(BaseModel):
    id: str
    label: str
    dtype: str
    is_protected: bool
    risk_level: str     # none | low | medium | high | critical


class GraphEdge(BaseModel):
    source: str
    target: str
    weight: float


# ---------------------------------------------------------------------------
# Fairness metrics (Friedler 2019 / Verma & Rubin 2018 taxonomy)
# ---------------------------------------------------------------------------

class GroupMetrics(BaseModel):
    group_value: str
    size: int
    base_rate: float          # P(Y=1 | group) — TRUE label rate in data
    prediction_rate: float = 0.0  # P(Ŷ=1 | group) — model prediction rate
    tpr: float                # True positive rate (sensitivity)
    fpr: float                # False positive rate
    precision: float          # Positive predictive value
    accuracy: float


class FairnessMetrics(BaseModel):
    protected_attribute: str
    outcome_column: str
    privileged_group: str
    positive_outcome: str
    # Aggregate metrics
    statistical_parity_diff: float       # P(Y=1|unpriv) - P(Y=1|priv)
    disparate_impact_ratio: float        # P(Y=1|unpriv) / P(Y=1|priv)
    equal_opportunity_diff: float        # TPR_unpriv - TPR_priv
    average_odds_diff: float             # (TPR_diff + FPR_diff) / 2
    predictive_parity_diff: float        # PPV_unpriv - PPV_priv
    model_accuracy_overall: float
    group_metrics: Dict[str, GroupMetrics]


# ---------------------------------------------------------------------------
# Calibration audit (Chouldechova 2017 / Guo et al. 2017)
# ---------------------------------------------------------------------------

class CalibrationBin(BaseModel):
    bin_lower: float
    bin_upper: float
    confidence: float   # mean predicted probability in bin
    accuracy: float     # fraction of positives in bin
    count: int


class GroupCalibration(BaseModel):
    group_value: str
    ece: float                  # Expected Calibration Error
    bins: List[CalibrationBin]
    max_calibration_gap: float  # max |accuracy - confidence| across bins


class CalibrationAudit(BaseModel):
    protected_attribute: str
    outcome_column: str
    group_calibration: Dict[str, GroupCalibration]
    calibration_gap: float      # max(ECE) - min(ECE) across groups
    is_calibrated: bool         # calibration_gap < 0.05


# ---------------------------------------------------------------------------
# Intersectional audit (Kearns 2018 fairness gerrymandering)
# ---------------------------------------------------------------------------

class IntersectionalGroup(BaseModel):
    group_key: str              # e.g. "race=Black,sex=Female"
    size: int
    base_rate: float            # P(Y=1 | intersection)
    spd_vs_privileged: float    # base_rate - privileged_base_rate


class IntersectionalAudit(BaseModel):
    protected_attributes: List[str]
    outcome_column: str
    privileged_combo: str       # e.g. "race=White,sex=Male"
    privileged_base_rate: float
    groups: List[IntersectionalGroup]
    max_spd_gap: float          # worst-case intersectional disparity
    flagged_groups: List[str]   # groups with |spd| > 0.1


# ---------------------------------------------------------------------------
# Reweighing result (Kamiran & Calders 2012)
# ---------------------------------------------------------------------------

class ReweighResult(BaseModel):
    protected_attribute: str
    outcome_column: str
    disc_before: float          # P(Y=1|priv) - P(Y=1|unpriv) raw
    disc_after: float           # discrimination after reweighing (should ≈ 0)
    n_samples: int


# ---------------------------------------------------------------------------
# Post-fix comparison
# ---------------------------------------------------------------------------

class MetricDelta(BaseModel):
    metric: str
    before: float
    after: float
    delta: float        # after - before (negative = improvement for SPD/AOD)
    improved: bool


class FixMetricsComparison(BaseModel):
    removed_feature: str
    deltas: List[MetricDelta]


# ---------------------------------------------------------------------------
# Conjunctive proxy (Zliobaite 2015 Type 2)
# ---------------------------------------------------------------------------

class ConjunctiveProxy(BaseModel):
    feature_a: str
    feature_b: str
    joint_skill: float
    skill_a: float
    skill_b: float
    interaction_gain: float   # joint_skill - max(skill_a, skill_b)
    protected_attribute: str
    risk_label: str


# ---------------------------------------------------------------------------
# Audit response
# ---------------------------------------------------------------------------

class AuditResponse(BaseModel):
    session_id: str
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    chains: List[Chain]
    summary: str
    fairness_metrics: List[FairnessMetrics] = []
    mitigated_fairness_metrics: List[FairnessMetrics] = []   # post-reweighing
    conjunctive_proxies: List[ConjunctiveProxy] = []
    calibration_audit: Optional[CalibrationAudit] = None
    intersectional_audit: Optional[IntersectionalAudit] = None


# ---------------------------------------------------------------------------
# Fix
# ---------------------------------------------------------------------------

class FixRequest(BaseModel):
    session_id: str
    chain_id: str
    fix_strategy: str = "drop"  # "drop" | "reweigh"


class ShapEntry(BaseModel):
    feature: str
    before: float
    after: float


class FixResponse(BaseModel):
    session_id: str
    chain_id: str
    removed_feature: str
    shap_values: List[ShapEntry]
    success: bool
    message: str
    metrics_comparison: Optional[FixMetricsComparison] = None
    reweigh_result: Optional[ReweighResult] = None


# ---------------------------------------------------------------------------
# Chat / Report
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str


class ReportRequest(BaseModel):
    session_id: str


class ReportResponse(BaseModel):
    download_url: str
