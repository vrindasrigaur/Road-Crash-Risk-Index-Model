import argparse
import os
import pandas as pd

from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

from xgboost import XGBRegressor

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

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
        "Support Vector Regression": SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.1),
        "XGBoost (XGBRegressor)": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1
        ),
    }

    cv = GroupKFold(n_splits=5)
    rows = []
    for name, model in models.items():
        pipe = Pipeline([('preprocess', preprocess), ('model', model)])
        pred = cross_val_predict(pipe, X, y, cv=cv, groups=groups, n_jobs=1)
        m = regression_metrics(y, pred)
        rows.append({
            'Model': name,
            'R2': round(m['r2'], 4),
            'MAE': round(m['mae'], 4),
            'RMSE': round(m['rmse'], 4),
            'RMS': round(m['rms'], 4),
        })

    out_df = pd.DataFrame(rows).sort_values('R2', ascending=False)
    out_path = os.path.join(args.out, 'model_comparison_metrics.xlsx')
    out_df.to_excel(out_path, index=False)

    print("\n=== MODEL COMPARISON (GroupKFold by Chainage) ===")
    print(out_df.to_string(index=False))
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
