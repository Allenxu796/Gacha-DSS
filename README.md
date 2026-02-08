# Gacha-DSS

Gacha-DSS is an intelligent decision-support system built on Monte Carlo simulation. It delivers a full closed loop from **simulation → statistical validation → training → risk-utility evaluation → decision output → RL evolution**.

## Highlights
- Industrial-grade project structure, supporting growth from ML to RL
- Reproducible simulation validation and distribution visualization
- Two-stage probability modeling: `P(target) = P(five_star) * P(target | five_star)`
- Risk-averse utility functions driving decision scores

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Raw Simulation Logs
```bash
python scripts/run_sim.py \
  --config configs/game_rules.yaml \
  --pulls 100000 \
  --output data/raw
```

### 3. Build Feature Data
```bash
python -m src.models.feature_factory \
  --input data/raw/<your_filename>.csv \
  --output data/processed/train.csv
```

### 4. Train Two-Stage Models
```bash
python scripts/train_pipeline.py \
  --data data/processed/train.csv \
  --model two_stage
```

### 5. Generate Decision Report (with Calibration Buckets)
```bash
python scripts/decision_report.py \
  --data data/processed/train.csv \
  --stageA artifacts/stageA_random_forest_model.joblib,artifacts/stageA_gbdt_model.joblib \
  --stageB artifacts/stageB_random_forest_model.joblib,artifacts/stageB_gbdt_model.joblib \
  --risk 0,0.5,1,2,3 \
  --out artifacts/decision_report.json
```

### 6. RL Baseline (Q-learning)
```bash
python scripts/run_rl_baseline.py \
  --config configs/game_rules.yaml \
  --episodes 200 \
  --pity_bucket 5 \
  --out artifacts/rl_policy.json
```

Inspect the policy table:
```bash
python scripts/inspect_rl_policy.py \
  --input artifacts/rl_policy.json
```

## Structure Overview
See `docs/roadmap.md` and `docs/math_model.md`.

## License
Add as needed.
