"""
Generate results/demo_metrics_summary.md from pipeline artifacts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def main():
    metrics = pd.read_csv(RESULTS_DIR / "model_metrics.csv")
    best = metrics.sort_values("auc_roc", ascending=False).iloc[0]

    opt_path = RESULTS_DIR / "best_model_optimization_comparison.csv"
    opt_note = ""
    if opt_path.exists():
        opt = pd.read_csv(opt_path)
        baseline = opt[opt["version"] == "baseline_xgboost"].iloc[0]
        optimized = opt[opt["version"] == "optimized_xgboost"].iloc[0]
        opt_note = f"""
## Optimization comparison (XGBoost)

| Version | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---------|----------|-----------|--------|-----|---------|
| Baseline | {baseline['accuracy']:.3f} | {baseline['precision']:.3f} | {baseline['recall']:.3f} | {baseline['f1_score']:.3f} | {baseline['auc_roc']:.3f} |
| Optimized | {optimized['accuracy']:.3f} | {optimized['precision']:.3f} | {optimized['recall']:.3f} | {optimized['f1_score']:.3f} | {optimized['auc_roc']:.3f} |

**Inference uses baseline XGBoost** (higher holdout AUC).
"""

    synth_path = RESULTS_DIR / "synthetic_predictions.csv"
    synth_note = "Not generated yet."
    if synth_path.exists():
        synth = pd.read_csv(synth_path)
        diabetic = (synth["prediction"] == 1).sum()
        synth_note = f"{len(synth)} patients scored; {diabetic} flagged as diabetic."

    content = f"""# Demo Metrics Summary

Generated from pipeline artifacts in `results/`.

## Best model (holdout test set)

- **Model:** {best['model']} (baseline, used for inference)
- **Accuracy:** {best['accuracy']:.3f}
- **Precision:** {best['precision']:.3f}
- **Recall:** {best['recall']:.3f}
- **F1-Score:** {best['f1_score']:.3f}
- **AUC-ROC:** {best['auc_roc']:.3f}

## All models

```
{metrics.to_string(index=False)}
```
{opt_note}
## Synthetic batch demo

{synth_note}

## How to reproduce

```bash
./run_pipeline.sh
streamlit run app/streamlit_app.py
```
"""

    out_path = RESULTS_DIR / "demo_metrics_summary.md"
    out_path.write_text(content, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
