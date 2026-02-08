from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect RL policy JSON.")
    parser.add_argument("--input", required=True, help="Path to rl_policy.json")
    args = parser.parse_args()

    policy = json.loads(Path(args.input).read_text(encoding="utf-8"))

    rows = []
    for k, action in policy.items():
        pity_bucket, guarantee, capture = [int(x) for x in k.split("|")]
        rows.append((pity_bucket, guarantee, capture, int(action)))

    rows.sort()

    print("pity_bucket\tguarantee\tcapture\taction")
    for r in rows:
        print(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}")


if __name__ == "__main__":
    main()
