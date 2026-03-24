import pandas as pd
import numpy as np
import warnings
import random

# ==============================
# 🔥 掛載 Google Drive（讓 Colab 可以讀取雲端資料）
# ==============================
from google.colab import drive
drive.mount('/content/drive')

# ==============================
# 🔥 安裝必要套件（Colab 預設沒有）
# ==============================
!pip install lightgbm xgboost optuna

# 固定隨機種子（讓結果可重現）
random.seed(42)

# ==============================
# 🔥 機器學習套件
# ==============================
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, StackingClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

import optuna

warnings.filterwarnings('ignore')

# ==============================
# 🔥 讀取資料
# ==============================
data_path = "/content/drive/MyDrive/資料科學與機器學習/boy or girl/"

train_df = pd.read_csv(data_path + "boy or girl 2025 train_missingValue.csv")
test_df = pd.read_csv(data_path + "boy or girl 2025 test no ans_missingValue.csv")

# 將 yt 欄位轉為數值（避免文字或錯誤格式）
train_df['yt'] = pd.to_numeric(train_df['yt'], errors='coerce')
test_df['yt'] = pd.to_numeric(test_df['yt'], errors='coerce')

# ==============================
# 🔥 移除低效特徵（經過實驗確認效果不好）
# ==============================
features_to_drop = ['self_intro', 'star_sign', 'phone_os']
train_df = train_df.drop(columns=features_to_drop)
test_df = test_df.drop(columns=features_to_drop)

# 分離特徵與標籤
X = train_df.drop(columns=['id', 'gender'])
y = train_df['gender']

X_test_submission = test_df.drop(columns=['id', 'gender'])
test_ids = test_df['id']

# ==============================
# 🔥 Missing Indicator（非常重要）
# ==============================
# 目的：讓模型知道「這個值原本是缺的」
for col in ['weight', 'height', 'yt']:
    X[f'{col}_is_missing'] = X[col].isnull().astype(int)
    X_test_submission[f'{col}_is_missing'] = X_test_submission[col].isnull().astype(int)

# ==============================
# 🔥 Outlier 處理
# ==============================
# 將極端異常值轉為 NaN，交給 imputer 處理
for col in X.columns:
    if X[col].dtype in ['float64', 'int64']:
        X[col] = X[col].apply(lambda val: np.nan if pd.notnull(val) and val > 1e15 else val)
        X_test_submission[col] = X_test_submission[col].apply(lambda val: np.nan if pd.notnull(val) and val > 1e15 else val)

# ==============================
# 🔥 Label Encoding（將 gender 轉為數字）
# ==============================
le_y = LabelEncoder()
y_encoded = le_y.fit_transform(y)

# ==============================
# 🔥 Iterative Imputer（用 RF 補值）
# ==============================
rf_imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    max_iter=10,
    random_state=42
)

# 訓練 + 補值
X_imputed = pd.DataFrame(rf_imputer.fit_transform(X), columns=X.columns)
X_test_imputed = pd.DataFrame(rf_imputer.transform(X_test_submission), columns=X_test_submission.columns)

# ==============================
# 🔥 Feature Engineering
# ==============================
def create_advanced_features(df):
    df_new = df.copy()

    # 避免 height = 0
    df_new['height'] = df_new['height'].clip(lower=1)

    # BMI（結合兩個特徵）
    df_new['BMI'] = df_new['weight'] / ((df_new['height'] / 100) ** 2)

    # log 轉換（減少偏態）
    df_new['yt_log'] = np.log1p(df_new['yt'].clip(lower=0))
    df_new['fb_friends_log'] = np.log1p(df_new['fb_friends'].clip(lower=0))

    # 移除原始欄位（避免重複資訊）
    df_new = df_new.drop(columns=['yt', 'fb_friends'])

    return df_new

X_imputed_eng = create_advanced_features(X_imputed)
X_test_imputed_eng = create_advanced_features(X_test_imputed)

# ==============================
# 🔥 Feature Selection（移除無效特徵）
# ==============================
useless_features = ['BMI', 'yt_log', 'fb_friends_log', 'weight_is_missing']

X_pruned = X_imputed_eng.drop(columns=useless_features, errors='ignore')
X_test_pruned = X_test_imputed_eng.drop(columns=useless_features, errors='ignore')

# ==============================
# 🔥 Scaling（讓數值範圍一致）
# ==============================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_pruned)
X_test_scaled = scaler.transform(X_test_pruned)

# ==============================
# 🔥 Train / Validation Split
# ==============================
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y_encoded,
    test_size=0.15,
    stratify=y_encoded,
    random_state=42
)

# ==============================
# 🔥 Optuna 自動調參
# ==============================
def objective(trial):

    # XGBoost 參數搜尋
    xgb = XGBClassifier(
        max_depth=trial.suggest_int("xgb_max_depth", 3, 6),
        learning_rate=trial.suggest_float("xgb_lr", 0.01, 0.1),
        n_estimators=trial.suggest_int("xgb_n", 100, 400),
        subsample=trial.suggest_float("xgb_sub", 0.7, 1.0),
        colsample_bytree=trial.suggest_float("xgb_col", 0.7, 1.0),
        eval_metric='logloss',
        random_state=42
    )

    # LightGBM 參數搜尋
    lgbm = LGBMClassifier(
        num_leaves=trial.suggest_int("lgbm_leaves", 20, 50),
        learning_rate=trial.suggest_float("lgbm_lr", 0.01, 0.1),
        n_estimators=trial.suggest_int("lgbm_n", 100, 400),
        random_state=42,
        verbose=-1
    )

    rf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)

    # Stacking 模型
    model = StackingClassifier(
        estimators=[('xgb', xgb), ('lgbm', lgbm), ('rf', rf)],
        final_estimator=LogisticRegression(),
        cv=5
    )

    # 使用 CV 評估
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()
    return score

# 開始搜尋最佳參數
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

best_params = study.best_params

# ==============================
# 🔥 建立最佳模型
# ==============================
xgb_clf = XGBClassifier(
    max_depth=best_params["xgb_max_depth"],
    learning_rate=best_params["xgb_lr"],
    n_estimators=best_params["xgb_n"],
    subsample=best_params["xgb_sub"],
    colsample_bytree=best_params["xgb_col"],
    eval_metric='logloss',
    random_state=42
)

lgbm_clf = LGBMClassifier(
    num_leaves=best_params["lgbm_leaves"],
    learning_rate=best_params["lgbm_lr"],
    n_estimators=best_params["lgbm_n"],
    random_state=42,
    verbose=-1
)

rf_clf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)

# 最終 Stacking 模型
ensemble_model = StackingClassifier(
    estimators=[
        ('XGBoost', xgb_clf),
        ('LightGBM', lgbm_clf),
        ('RandomForest', rf_clf)
    ],
    final_estimator=LogisticRegression(),
    cv=5
)

# ==============================
# 🔥 Cross Validation 評估
# ==============================
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(ensemble_model, X_train, y_train, cv=skf, scoring='accuracy')

print(f"CV 平均準確率: {cv_scores.mean():.4f}")

# ==============================
# 🔥 Validation 評估
# ==============================
ensemble_model.fit(X_train, y_train)
y_val_pred = ensemble_model.predict(X_val)

val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# ==============================
# 🔥 預測測試集
# ==============================
test_predictions_encoded = ensemble_model.predict(X_test_scaled)
test_predictions_original = le_y.inverse_transform(test_predictions_encoded)

# ==============================
# 🔥 輸出 submission
# ==============================
submission_df = pd.DataFrame({
    'id': test_ids,
    'gender': test_predictions_original
})

submission_filename = data_path + "submission_stacking_optuna.csv"
submission_df.to_csv(submission_filename, index=False)

print(f"已輸出：{submission_filename}")