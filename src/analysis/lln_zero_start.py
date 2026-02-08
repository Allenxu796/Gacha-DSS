from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import math

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from ..simulation.engine import GachaEngine, State, config_from_dict


@dataclass
class LLNStats:
    total_pulls: int
    five_star_rate: float
    ci_low: float
    ci_high: float


def _wilson_interval(p_hat: float, n: int, z: float = 1.96) -> (float, float):
    if n == 0:
        return 0.0, 1.0
    denom = 1 + (z ** 2) / n
    center = (p_hat + (z ** 2) / (2 * n)) / denom
    margin = (z / denom) * math.sqrt((p_hat * (1 - p_hat) / n) + (z ** 2) / (4 * n ** 2))
    return max(0.0, center - margin), min(1.0, center + margin)


def _load_config(path: str) -> Dict:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to load YAML configs. Install with: pip install pyyaml"
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_zero_start_validation(raw_config: Dict, n_pulls: Optional[int] = None) -> LLNStats:
    sim_config = config_from_dict(raw_config)
    engine = GachaEngine(sim_config)
    engine.state = State()  # force zero-start state

    if n_pulls is None:
        n_pulls = int(raw_config.get("validation", {}).get("min_samples", 100000))

    results = engine.run(n_pulls)
    total_pulls = len(results)
    five_star_count = sum(1 for r in results if r.is_five_star)
    five_star_rate = five_star_count / total_pulls if total_pulls else 0.0
    ci_low, ci_high = _wilson_interval(five_star_rate, total_pulls)

    return LLNStats(
        total_pulls=total_pulls,
        five_star_rate=five_star_rate,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="LLN validation from zero-start pity only."
    )
    parser.add_argument("--config", required=True, help="Path to game_rules.yaml")
    parser.add_argument("--pulls", type=int, default=None, help="Number of pulls to simulate")
    args = parser.parse_args()

    raw_config = _load_config(args.config)
    stats = run_zero_start_validation(raw_config, n_pulls=args.pulls)

    expected = raw_config.get("validation", {}).get("expected_overall_five_star_rate")
    tolerance = raw_config.get("validation", {}).get("tolerance")

    print("=== Zero-Start Validation Summary ===")
    print(f"Total pulls: {stats.total_pulls}")
    print(f"Five-star rate: {stats.five_star_rate:.6f}")
    print(f"Wilson 95% CI: [{stats.ci_low:.6f}, {stats.ci_high:.6f}]")
    if expected is not None and tolerance is not None:
        low = expected - tolerance
        high = expected + tolerance
        status = "PASS" if low <= stats.five_star_rate <= high else "FAIL"
        print(f"Expected range: [{low:.6f}, {high:.6f}] -> {status}")


if __name__ == "__main__":
    main()
