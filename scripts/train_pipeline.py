from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.ml_agent import (
    load_dataset,
    select_features,
    train_random_forest,
    train_gbdt,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML model for Gacha decision.")
    parser.add_argument("--data", required=True, help="Path to processed dataset CSV")
    parser.add_argument("--label", default="label_is_target", help="Label column name")
    parser.add_argument("--model", choices=["random_forest", "gbdt", "both", "two_stage"], default="random_forest")
    parser.add_argument("--out_dir", default="artifacts", help="Output directory")
    args = parser.parse_args()

    df = load_dataset(args.data)
    df, label_col = select_features(df, args.label)
    if args.model != "two_stage":
        if label_col == "label_is_target" and "label_is_five_star" in df.columns:
            df = df[df["label_is_five_star"] == 1].copy()
            df.drop(columns=["label_is_five_star"], inplace=True)
    drop_cols = [c for c in ["pull_index"] if c in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

    results = []
    if args.model in ("random_forest", "both"):
        results.append(train_random_forest(df, label_col))
    if args.model in ("gbdt", "both"):
        results.append(train_gbdt(df, label_col))

    if args.model == "two_stage":
        # Stage A: predict five-star on all samples
        if "label_is_five_star" not in df.columns:
            raise RuntimeError("label_is_five_star not found in dataset for two_stage")
        df_stage_a = df.drop(columns=["label_is_target"]).copy()
        df_stage_a_label = "label_is_five_star"
        rfa = train_random_forest(df_stage_a, df_stage_a_label)
        rfa.model_name = "stageA_random_forest"
        results.append(rfa)
        gbda = train_gbdt(df_stage_a, df_stage_a_label)
        gbda.model_name = "stageA_gbdt"
        results.append(gbda)

        # Stage B: predict target conditional on five-star
        df_stage_b = df[df["label_is_five_star"] == 1].copy()
        df_stage_b.drop(columns=["label_is_five_star"], inplace=True)
        rfb = train_random_forest(df_stage_b, "label_is_target")
        rfb.model_name = "stageB_random_forest"
        results.append(rfb)
        gbdb = train_gbdt(df_stage_b, "label_is_target")
        gbdb.model_name = "stageB_gbdt"
        results.append(gbdb)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        model_path = out_dir / f"{result.model_name}_model.joblib"
        metrics_path = out_dir / f"{result.model_name}_metrics.json"
        fi_path = out_dir / f"{result.model_name}_feature_importance.json"

        joblib.dump(result.model, model_path)
        metrics_path.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
        fi_path.write_text(json.dumps(result.feature_importance, indent=2), encoding="utf-8")

        print(f"Saved model to: {model_path}")
        print(f"Saved metrics to: {metrics_path}")
        print(f"Saved feature importance to: {fi_path}")


if __name__ == "__main__":
    main()
