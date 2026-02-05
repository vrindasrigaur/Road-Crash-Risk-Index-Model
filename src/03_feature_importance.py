import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from utils import load_and_clean_excel, detect_target_column, build_preprocessor, safe_drop_cols

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='Path to Excel dataset')
    ap.add_argument('--out', default='outputs', help='Output folder')
    ap.add_argument('--top', type=int, default=10, help='Top N parameters to plot')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = load_and_clean_excel(args.data)
    target_col = detect_target_column(df)

    drop_cols = ['Timestamp', 'Name', 'Age', 'Year of Survey (As per on Video)']
    X = safe_drop_cols(df, drop_cols + [target_col])
    y = df[target_col].astype(float)

    preprocess, cat_cols, num_cols = build_preprocessor(X)

    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=1)
    pipe = Pipeline([('preprocess', preprocess), ('model', rf)])
    pipe.fit(X, y)

    pre = pipe.named_steps['preprocess']
    model = pipe.named_steps['model']

    cat_features = pre.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols)
    num_features = np.array(num_cols, dtype=str)
    feature_names = np.concatenate([cat_features, num_features])

    enc_imp = pd.DataFrame({
        'Encoded_Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    def original_name(encoded):
        s = str(encoded)
        for col in (cat_cols + num_cols):
            if s.startswith(str(col)):
                return col
        return s

    enc_imp['Original_Parameter'] = enc_imp['Encoded_Feature'].apply(original_name)

    agg = (enc_imp.groupby('Original_Parameter', as_index=False)['Importance'].sum()
           .sort_values('Importance', ascending=False))
    agg['Normalized_Weight'] = agg['Importance'] / agg['Importance'].sum()

    out_xlsx = os.path.join(args.out, 'feature_importance.xlsx')
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
        enc_imp.to_excel(writer, index=False, sheet_name='Encoded_Features')
        agg.to_excel(writer, index=False, sheet_name='Aggregated_By_Parameter')

    topn = agg.head(args.top).iloc[::-1]  # reverse for barh
    plt.figure(figsize=(10, 6))
    plt.barh(topn['Original_Parameter'], topn['Normalized_Weight'])
    plt.xlabel('Normalized Importance Weight')
    plt.title(f'Top {args.top} Contributing Factors to Crash Risk Index (Random Forest)')
    plt.tight_layout()

    out_png = os.path.join(args.out, f'feature_importance_top{args.top}.png')
    plt.savefig(out_png, dpi=300)
    plt.show()

    print(f"Saved: {out_xlsx}")
    print(f"Saved: {out_png}")
    print("\\nTop parameters:")
    print(agg.head(args.top).to_string(index=False))

if __name__ == "__main__":
    main()
