from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation.engine import config_from_dict
from src.models.rl_env import GachaEnv, EnvConfig
from src.models.rl_baseline import QConfig, q_learn, derive_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Q-learning baseline and save policy.")
    parser.add_argument("--config", required=True, help="Path to game_rules.yaml")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--pity_bucket", type=int, default=5)
    parser.add_argument("--out", default="artifacts/rl_policy.json")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    env = GachaEnv(config_from_dict(cfg), EnvConfig())

    qcfg = QConfig(episodes=args.episodes, pity_bucket=args.pity_bucket)
    q = q_learn(env, qcfg)
    policy = derive_policy(q)
    policy_serializable = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in policy.items()}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(policy_serializable, indent=2), encoding="utf-8")

    print(f"Saved policy to: {out_path}")


if __name__ == "__main__":
    main()
