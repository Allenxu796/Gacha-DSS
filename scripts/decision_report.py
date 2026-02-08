from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.utility_func import UtilityConfig, summarize_decision


def load_model(path: str):
    obj = joblib.load(path)
    # Backward compatibility: if a TrainResult was saved, extract the model.
    if hasattr(obj, "model"):
        return obj.model
    return obj


def predict_prob(model_obj, X: pd.DataFrame) -> pd.Series:
    return pd.Series(model_obj.predict_proba(X)[:, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate decision report from two-stage models.")
    parser.add_argument("--data", required=True, help="Processed dataset CSV")
    parser.add_argument(
        "--stageA",
        required=True,
        help="Stage A model(s) joblib. Comma-separated for fusion.",
    )
    parser.add_argument(
        "--stageB",
        required=True,
        help="Stage B model(s) joblib. Comma-separated for fusion.",
    )
    parser.add_argument(
        "--risk",
        default="0,0.5,1,2",
        help="Comma-separated risk aversion coefficients",
    )
    parser.add_argument("--out", default="artifacts/decision_report.json", help="Output report path")
    parser.add_argument(
        "--bucket",
        default="0-10,11-20,21-30,31-40,41-50,51-60,61-70,71-80,81-90",
        help="Pity bucket ranges like 0-10,11-20,...",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    X = df.drop(columns=["label_is_five_star", "label_is_target", "pull_index"], errors="ignore")

    stage_a_paths = [p.strip() for p in args.stageA.split(",") if p.strip()]
    stage_b_paths = [p.strip() for p in args.stageB.split(",") if p.strip()]
    if not stage_a_paths or not stage_b_paths:
        raise RuntimeError("stageA and stageB must contain at least one model path")

    models_a = [load_model(p) for p in stage_a_paths]
    models_b = [load_model(p) for p in stage_b_paths]

    probs_a = [predict_prob(m, X) for m in models_a]
    p_five = sum(probs_a) / len(probs_a)

    df_five = df[df["label_is_five_star"] == 1].copy()
    X_five = df_five.drop(columns=["label_is_five_star", "label_is_target", "pull_index"], errors="ignore")
    probs_b = [predict_prob(m, X_five) for m in models_b]
    p_target_given = sum(probs_b) / len(probs_b)

    prob_five_star = float(p_five.mean())
    prob_target_given_five = float(p_target_given.mean())

    risks = [float(r) for r in args.risk.split(",") if r.strip() != ""]
    summary = {
        "prob_five_star": prob_five_star,
        "prob_target_given_five": prob_target_given_five,
        "prob_target": prob_five_star * prob_target_given_five,
        "risk_curve": [],
    }
    for r in risks:
        line = summarize_decision(prob_five_star, prob_target_given_five, UtilityConfig(r))
        line["risk_aversion"] = r
        summary["risk_curve"].append(line)

    # Bucket report by pity_before
    buckets = []
    for token in args.bucket.split(","):
        token = token.strip()
        if not token:
            continue
        lo, hi = token.split("-")
        buckets.append((int(lo), int(hi)))

    bucket_rows = []
    for lo, hi in buckets:
        sub = df[(df["pity_before"] >= lo) & (df["pity_before"] <= hi)]
        if len(sub) == 0:
            continue
        X_sub = sub.drop(columns=["label_is_five_star", "label_is_target", "pull_index"], errors="ignore")
        p_five_sub = sum(predict_prob(m, X_sub) for m in models_a) / len(models_a)

        sub_five = sub[sub["label_is_five_star"] == 1]
        if len(sub_five) == 0:
            p_target_given = 0.0
            empirical_target_given = 0.0
        else:
            X_sub_five = sub_five.drop(
                columns=["label_is_five_star", "label_is_target", "pull_index"], errors="ignore"
            )
            p_target_given = float(
                (sum(predict_prob(m, X_sub_five) for m in models_b) / len(models_b)).mean()
            )
            empirical_target_given = float(sub_five["label_is_target"].mean())

        empirical_five = float(sub["label_is_five_star"].mean())

        bucket_rows.append(
            {
                "bucket": f"{lo}-{hi}",
                "samples": int(len(sub)),
                "prob_five_star_pred": float(p_five_sub.mean()),
                "prob_five_star_empirical": empirical_five,
                "prob_target_given_five_pred": p_target_given,
                "prob_target_given_five_empirical": empirical_target_given,
                "prob_target_pred": float(p_five_sub.mean()) * p_target_given,
                "prob_target_empirical": empirical_five * empirical_target_given,
            }
        )

    summary["bucket_report"] = bucket_rows

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved decision report to: {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
