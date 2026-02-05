# Road Segment Crash Risk Index (CRI) - Full ML Package + XGBoost (Regression, 500m)

This package runs:
- Multi-model regression comparison (Linear Regression, Random Forest, Gradient Boosting, SVR, **XGBoost**)
- Random Forest training + 500m segment CRI aggregation
- Feature importance table + Top-N plot (Random Forest)

## VS Code setup
```bash
python -m venv .venv
```
Activate:
- Windows: `.\.venv\Scripts\activate`
- macOS/Linux: `source .venv/bin/activate`

Install:
```bash
pip install -r requirements.txt
```

## Dataset
`data/Expert_Opinion_Survey_Responses.xlsx`

## Run
```bash
python src/01_model_comparison.py --data data/Expert_Opinion_Survey_Responses.xlsx --out outputs
python src/02_train_rf_and_segment_cri.py --data data/Expert_Opinion_Survey_Responses.xlsx --out outputs
python src/03_feature_importance.py --data data/Expert_Opinion_Survey_Responses.xlsx --out outputs --top 10
```

Outputs go to `outputs/`.
