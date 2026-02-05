@echo off
python src\01_model_comparison.py --data data\Expert_Opinion_Survey_Responses.xlsx --out outputs
python src\02_train_rf_and_segment_cri.py --data data\Expert_Opinion_Survey_Responses.xlsx --out outputs
python src\03_feature_importance.py --data data\Expert_Opinion_Survey_Responses.xlsx --out outputs --top 10
pause
