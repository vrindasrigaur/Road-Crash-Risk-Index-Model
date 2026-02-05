

import argparse
import os
import json
import pandas as pd

from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from utils import load_and_clean_excel, detect_target_column, build_preprocessor, safe_drop_cols, regression_metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='Path to Excel dataset')
    ap.add_argument('--out', default='outputs', help='Output folder')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = load_and_clean_excel(args.data)
    target_col = detect_target_column(df)
    group_col = 'Chainage'
    if group_col not in df.columns:
        raise ValueError("Expected 'Chainage' column for 500m segment grouping.")

    drop_cols = ['Timestamp', 'Name', 'Age', 'Year of Survey (As per on Video)']
    X = safe_drop_cols(df, drop_cols + [target_col])
    y = df[target_col].astype(float)
    groups = df[group_col].astype(str)

    preprocess, _, _ = build_preprocessor(X)

    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=1)
    pipe = Pipeline([('preprocess', preprocess), ('model', rf)])

    # Honest metrics via out-of-fold predictions
    cv = GroupKFold(n_splits=5)
    oof_pred = cross_val_predict(pipe, X, y, cv=cv, groups=groups, n_jobs=1)
    metrics = regression_metrics(y, oof_pred)

    metrics_path = os.path.join(args.out, 'rf_oof_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print("=== Random Forest (OOF, GroupKFold by Chainage) ===")
    print({k: round(v, 4) for k, v in metrics.items()})
    print(f"Saved: {metrics_path}")

    # Fit on full data for final predictions
    pipe.fit(X, y)
    full_pred = pipe.predict(X)

    row_pred = pd.DataFrame({
        'Chainage': groups,
        'Actual_CrashIndex': y.values,
        'Predicted_CRI': full_pred
    })
    row_out = os.path.join(args.out, 'rf_row_predictions.xlsx')
    row_pred.to_excel(row_out, index=False)

    seg = (row_pred.groupby('Chainage', as_index=False)
           .agg(
               n_responses=('Predicted_CRI', 'size'),
               actual_mean=('Actual_CrashIndex', 'mean'),
               predicted_mean=('Predicted_CRI', 'mean'),
               predicted_std=('Predicted_CRI', 'std')
           )
           .sort_values('predicted_mean', ascending=False))

    seg_out = os.path.join(args.out, 'segment_cri_500m.xlsx')
    seg.to_excel(seg_out, index=False)

    print(f"Saved: {row_out}")
    print(f"Saved: {seg_out}")

if __name__ == "__main__":
    main()
