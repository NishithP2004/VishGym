"""Reproducible, review-gated training utilities for VishGym."""

from vishgym.training.evaluation import EvaluationReport, PromotionDecision, promotion_decision
from vishgym.training.dataset import export_warm_start_dataset, load_training_examples
from vishgym.training.synthetic_data import build_warm_start_examples

__all__ = [
    "EvaluationReport",
    "PromotionDecision",
    "build_warm_start_examples",
    "export_warm_start_dataset",
    "load_training_examples",
    "promotion_decision",
]
