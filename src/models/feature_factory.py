from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv


@dataclass
class FeatureRow:
    pull_index: int
    pity_before: int
    guarantee_before: int
    capture_counter_before: int
    is_five_star: int
    is_target: int


def transform_row(row: dict) -> FeatureRow:
    return FeatureRow(
        pull_index=int(row["pull_index"]),
        pity_before=int(row["pity_before"]),
        guarantee_before=int(row["guarantee_before"]),
        capture_counter_before=int(row["capture_counter_before"]),
        is_five_star=int(row["is_five_star"]),
        is_target=int(row["is_target"]),
    )


def build_features(input_path: str, output_path: str) -> Path:
    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [transform_row(r) for r in reader]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "pull_index",
                "pity_before",
                "guarantee_before",
                "capture_counter_before",
                "label_is_five_star",
                "label_is_target",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.pull_index,
                    r.pity_before,
                    r.guarantee_before,
                    r.capture_counter_before,
                    r.is_five_star,
                    r.is_target,
                ]
            )

    return out_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Transform raw simulation logs to feature dataset.")
    parser.add_argument("--input", required=True, help="Path to raw CSV log")
    parser.add_argument("--output", required=True, help="Path to processed CSV")
    args = parser.parse_args()

    out_path = build_features(args.input, args.output)
    print(f"Wrote processed dataset to: {out_path}")


if __name__ == "__main__":
    main()
