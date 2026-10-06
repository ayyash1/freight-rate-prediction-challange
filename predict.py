"""
Prediction and evaluation runner for Freight Rate ML Pipeline.
Generates:
1. validation_predictions.csv (12,000 rows)
2. data/december_chart_inputs.csv (31 rows completed)
3. Executes score.py to validate predictions and produce candidate_december.png
"""
from __future__ import annotations
import sys
import subprocess
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from src.pipeline import FreightRatePipeline


def main():
    parser = argparse.ArgumentParser(description="Generate Predictions for Freight Rate Challenge")
    parser.add_argument("--data-dir", default="data", help="Directory containing raw data files")
    parser.add_argument("--output-csv", default="validation_predictions.csv", help="Output predictions path")
    parser.add_argument("--scorer-dir", default="scorer_results", help="Directory for scorer chart")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_csv = Path(args.output_csv)
    scorer_dir = Path(args.scorer_dir)
    scorer_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FREIGHT RATE PREDICTION CHALLENGE - INFERENCE & SCORING PIPELINE")
    print("=" * 70)

    # 1. Load Data
    train_path = data_dir / "train_test.csv"
    val_path = data_dir / "validation.csv"
    template_path = data_dir / "validation_predictions_template.csv"
    dec_path = data_dir / "december_chart_inputs.csv"

    print(f"\n[1/4] Loading input datasets...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    template_df = pd.read_csv(template_path)
    dec_df = pd.read_csv(dec_path)

    print(f"  Training set: {len(train_df):,} loads")
    print(f"  Validation set: {len(val_df):,} loads")
    print(f"  Template: {len(template_df):,} rows")
    print(f"  December chart inputs: {len(dec_df):,} rows")

    # 2. Fit Full Production Pipeline
    print("\n[2/4] Fitting Production Multi-Model Ensemble...")
    pipeline = FreightRatePipeline(ensemble_weights={'lightgbm': 0.40, 'catboost': 0.40, 'xgboost': 0.20})
    pipeline.fit(train_df, val_df_for_coords=val_df)
    print("  Production Ensemble successfully fitted!")

    # 3. Generate Predictions on Validation Set
    print("\n[3/4] Generating predictions for 12,000 validation loads...")
    val_preds = pipeline.predict_validation(val_df, template_df=template_df)
    
    # Save to validation_predictions.csv
    val_preds.to_csv(output_csv, index=False)
    print(f"  Saved final predictions to: {output_csv}")
    print(f"  Shape: {val_preds.shape}")
    print(f"  Columns: {list(val_preds.columns)}")
    print(f"  Sample predictions:\n{val_preds.head(5).to_string(index=False)}")

    # 4. Generate Predictions for December Benchmark Chart
    print("\n[4/4] Generating predictions for 31 December benchmark loads (Lexington -> Fort Wayne)...")
    dec_preds = pipeline.predict_december_chart(dec_df, val_df=val_df)
    
    # Save completed December chart inputs
    dec_preds.to_csv(data_dir / "december_chart_inputs.csv", index=False)
    print(f"  Saved completed December predictions to: {data_dir / 'december_chart_inputs.csv'}")
    print(f"  Sample December predictions:\n{dec_preds.head(5).to_string(index=False)}")

    # 5. Run score.py Validation & Chart Generation
    print("\n" + "=" * 70)
    print("RUNNING score.py VERIFICATION & CHART GENERATION")
    print("=" * 70)
    cmd = [
        sys.executable,
        "score.py",
        "--predictions", str(output_csv),
        "--december-predictions", str(data_dir / "december_chart_inputs.csv"),
        "--output-dir", str(scorer_dir)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("Score.py output:")
    print(result.stdout)
    if result.stderr:
        print("Score.py stderr:", result.stderr)

    if result.returncode == 0:
        print("\nSUCCESS: All predictions validated and chart successfully generated!")
        print(f"Chart saved at: {scorer_dir / 'candidate_december.png'}")
    else:
        print(f"\nERROR: score.py failed with exit code {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
