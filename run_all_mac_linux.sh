#!/usr/bin/env bash
set -e
python3 src/01_model_comparison.py --data data/Expert_Opinion_Survey_Responses.xlsx --out outputs
python3 src/02_train_rf_and_segment_cri.py --data data/Expert_Opinion_Survey_Responses.xlsx --out outputs
python3 src/03_feature_importance.py --data data/Expert_Opinion_Survey_Responses.xlsx --out outputs --top 10
