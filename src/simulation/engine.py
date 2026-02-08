from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple
import random

from .rules_5_0 import apply_five_star_rule


@dataclass
class SimulationConfig:
    base_five_star_prob: float
    hard_pity: int
    soft_pity_start: int
    soft_pity_step: float
    target_prob_no_guarantee: float
    target_prob_guarantee: float
    capture_enabled: bool
    capture_hard: int
    capture_prob: float
    seed: Optional[int] = None
    soft_pity_mode: str = "linear"


@dataclass
class State:
    pulls_since_five_star: int = 0
    guarantee: bool = False
    capture_counter: int = 0
    total_pulls: int = 0
    total_five_stars: int = 0
    total_target_five_stars: int = 0


@dataclass
class PullResult:
    pity_before: int
    guarantee_before: bool
    capture_counter_before: int
    is_five_star: bool
    is_target: bool
    pity: int
    guarantee_after: bool
    capture_counter_after: int


class GachaEngine:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = random.Random(config.seed)

    def reset(self) -> None:
        self.state = State()

    def _five_star_probability(self, pity_count: int) -> float:
        if pity_count >= self.config.hard_pity:
            return 1.0
        if pity_count < self.config.soft_pity_start:
            return self.config.base_five_star_prob
        if getattr(self.config, "soft_pity_mode", "linear") == "quadratic":
            # Nonlinear soft pity curve: ease-in quadratic.
            span = max(1, self.config.hard_pity - self.config.soft_pity_start)
            t = (pity_count - self.config.soft_pity_start + 1) / span
            t = min(max(t, 0.0), 1.0)
            eased = t * t
            return self.config.base_five_star_prob + (1.0 - self.config.base_five_star_prob) * eased

        steps = pity_count - self.config.soft_pity_start + 1
        return min(1.0, self.config.base_five_star_prob + steps * self.config.soft_pity_step)

    def pull_once(self) -> PullResult:
        s = self.state
        pity_before = s.pulls_since_five_star
        guarantee_before = s.guarantee
        capture_counter_before = s.capture_counter
        s.total_pulls += 1
        s.pulls_since_five_star += 1

        prob = self._five_star_probability(s.pulls_since_five_star)
        is_five_star = self.rng.random() < prob

        is_target = False
        if is_five_star:
            s.total_five_stars += 1
            s.pulls_since_five_star = 0

            is_target, s.guarantee, s.capture_counter = apply_five_star_rule(
                rng=self.rng,
                guarantee=s.guarantee,
                capture_enabled=self.config.capture_enabled,
                capture_counter=s.capture_counter,
                capture_hard=self.config.capture_hard,
                capture_prob=self.config.capture_prob,
                p_target_no_guarantee=self.config.target_prob_no_guarantee,
                p_target_guarantee=self.config.target_prob_guarantee,
            )

            if is_target:
                s.total_target_five_stars += 1

        return PullResult(
            pity_before=pity_before,
            guarantee_before=guarantee_before,
            capture_counter_before=capture_counter_before,
            is_five_star=is_five_star,
            is_target=is_target,
            pity=s.pulls_since_five_star,
            guarantee_after=s.guarantee,
            capture_counter_after=s.capture_counter,
        )

    def run(self, n_pulls: int) -> List[PullResult]:
        if not hasattr(self, "state"):
            self.reset()
        results: List[PullResult] = []
        for _ in range(n_pulls):
            results.append(self.pull_once())
        return results


def config_from_dict(raw: Dict) -> SimulationConfig:
    base = raw["base_probability"]["five_star"]
    pity = raw["pity"]
    rate_up = raw["rate_up"]
    capture = raw.get("capture_mechanism", {})
    seed = raw.get("random", {}).get("seed")

    return SimulationConfig(
        base_five_star_prob=base,
        hard_pity=int(pity["hard_pity"]),
        soft_pity_start=int(pity["soft_pity_start"]),
        soft_pity_step=float(pity["soft_pity_step"]),
        soft_pity_mode=str(pity.get("soft_pity_mode", "linear")),
        target_prob_no_guarantee=float(rate_up["target_probability_when_no_guarantee"]),
        target_prob_guarantee=float(rate_up["target_probability_when_guarantee"]),
        capture_enabled=bool(capture.get("enabled", False)),
        capture_hard=int(capture.get("hard_capture", 0)),
        capture_prob=float(capture.get("capture_probability", 0.0)),
        seed=seed,
    )
