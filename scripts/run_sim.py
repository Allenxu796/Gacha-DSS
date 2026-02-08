from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation.engine import GachaEngine, config_from_dict


def run_sim(config_path: str, n_pulls: int, output_dir: str, seed_override: int | None) -> Path:
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    if seed_override is not None:
        raw_config.setdefault("random", {})["seed"] = seed_override

    engine = GachaEngine(config_from_dict(raw_config))
    engine.reset()
    results = engine.run(n_pulls)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"sim_raw_{stamp}.csv"

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "pull_index",
                "pity_before",
                "guarantee_before",
                "capture_counter_before",
                "is_five_star",
                "is_target",
                "pity",
                "guarantee_after",
                "capture_counter_after",
            ]
        )
        for i, r in enumerate(results, start=1):
            writer.writerow(
                [
                    i,
                    r.pity_before,
                    int(r.guarantee_before),
                    r.capture_counter_before,
                    int(r.is_five_star),
                    int(r.is_target),
                    r.pity,
                    int(r.guarantee_after),
                    r.capture_counter_after,
                ]
            )

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Monte Carlo simulations and persist raw logs.")
    parser.add_argument("--config", required=True, help="Path to game_rules.yaml")
    parser.add_argument("--pulls", type=int, default=100000, help="Number of pulls to simulate")
    parser.add_argument("--output", default="data/raw", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    args = parser.parse_args()

    out_path = run_sim(args.config, args.pulls, args.output, args.seed)
    print(f"Wrote raw simulation log to: {out_path}")


if __name__ == "__main__":
    main()
