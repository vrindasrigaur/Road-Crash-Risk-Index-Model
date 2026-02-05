import re
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def clean_checkbox_text(x):
    if pd.isna(x):
        return x
    if isinstance(x, str):
        x = re.sub(r'[□]', '', x)
        x = re.sub(r'\s+', ' ', x).strip()
        if x.lower() in ['yes', 'y', 'true']:
            return 'Yes'
        if x.lower() in ['no', 'n', 'false']:
            return 'No'
    return x

def load_and_clean_excel(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.map(clean_checkbox_text)
    return df

def detect_target_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if 'Overall Crash Index' in str(c)]
    if not candidates:
        raise ValueError("Target column not found. Expected a column containing 'Overall Crash Index'.")
    return candidates[0]

def safe_drop_cols(df: pd.DataFrame, cols):
    existing = [c for c in cols if c in df.columns]
    return df.drop(columns=existing)

def build_preprocessor(X: pd.DataFrame):
    cat_cols = [c for c in X.columns if X[c].dtype == 'object']
    num_cols = [c for c in X.columns if X[c].dtype != 'object']

    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocess = ColumnTransformer([
        ('num', numeric_pipe, num_cols),
        ('cat', categorical_pipe, cat_cols)
    ])

    return preprocess, cat_cols, num_cols

def regression_metrics(y_true, y_pred):
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    rms = float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))
    return {'r2': r2, 'mae': mae, 'rmse': rmse, 'rms': rms}
