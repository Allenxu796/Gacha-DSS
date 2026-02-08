from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import math


@dataclass
class UtilityConfig:
    risk_aversion: float  # Arrow-Pratt coefficient (a >= 0)


def expected_utility(prob_success: float, reward: float, cost: float, cfg: UtilityConfig) -> float:
    """
    CRRA-like utility with risk aversion on net outcome.
    U(x) = (1 - exp(-a * x)) / a, with a -> 0 => linear.
    """
    x_success = reward - cost
    x_fail = -cost

    if cfg.risk_aversion == 0:
        return prob_success * x_success + (1 - prob_success) * x_fail

    a = cfg.risk_aversion
    u_success = (1 - math.exp(-a * x_success)) / a
    u_fail = (1 - math.exp(-a * x_fail)) / a
    return prob_success * u_success + (1 - prob_success) * u_fail


def decision_score(prob_success: float, cfg: UtilityConfig) -> float:
    """
    Convenience score: utility under unit reward/cost.
    """
    return expected_utility(prob_success, reward=1.0, cost=1.0, cfg=cfg)


def summarize_decision(prob_five_star: float, prob_target_given_five: float, cfg: UtilityConfig) -> Dict[str, float]:
    prob_target = prob_five_star * prob_target_given_five
    score = decision_score(prob_target, cfg)
    return {
        "prob_five_star": prob_five_star,
        "prob_target_given_five": prob_target_given_five,
        "prob_target": prob_target,
        "utility_score": score,
    }
