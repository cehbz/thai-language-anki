from pydantic import BaseModel
from ..config import RulebookConfig
from ..core.findings import Dimension
from ..core.pipeline import EvalResult

class Scores(BaseModel):
    integrity: float
    language: float
    method: float
    content: float

def _deducted(result, cfg, dim) -> float:
    return sum(cfg.deductions.get(str(f.severity), 0)
               for f in result.findings if f.dimension == dim)

def compute_scores(result: EvalResult, cfg: RulebookConfig) -> Scores:
    def simple(dim):
        return max(0.0, 100.0 - _deducted(result, cfg, dim))
    weights = {m.name: cfg.metric_weights.get(m.name, 1.0) for m in result.metrics}
    if weights:
        blend = sum(cfg.metric_weights.get(m.name, 1.0) * m.value
                    for m in result.metrics) / sum(weights.values())
        method = min(100.0, max(0.0, 100.0 * blend
                                - _deducted(result, cfg, Dimension.METHOD)))
    else:
        method = 0.0
    return Scores(integrity=simple(Dimension.INTEGRITY),
                  language=simple(Dimension.LANGUAGE),
                  method=method,
                  content=simple(Dimension.CONTENT))
