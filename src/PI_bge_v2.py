# =============================================================
# PI_bge_v2.py — 基於 PI_bge.py 的改進版
#
# 相較 PI_bge.py 的改動：
#   1. 資料清洗：fb_friends 上限 10000→1000000, yt 上限 100→10000
#      原因：放寬上限以保留更多合理數值，避免過度截斷導致遺失相關性
#   2. 補值：median → MissForest (IterativeImputer + RandomForest)
#      原因：MissForest 利用特徵間關聯做多變量補值，比單純中位數更準確
#   3. 不平衡處理：class_weight → SMOTE 上採樣
#      原因：生成合成少數類樣本，讓模型學到更多少數類的決策邊界
#   4. 衍生特徵：移除 height_weight_ratio，只保留 BMI
#   5. 模型：單一 LightGBM → Stacking (LGB + XGB + RF → LR)
# =============================================================

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import lightgbm as lgb
from xgboost import XGBClassifier
import os

# 1. 載入資料
train = pd.read_csv('../data/boy_or_girl_2025_train_missingValue.csv')
test  = pd.read_csv('../data/boy_or_girl_2025_test_no_ans_missingValue.csv')

# 2. 資料清洗（fb_friends/yt 上限放寬，保留更多數值特徵的變異）
for df in [train, test]:
    df['yt'] = pd.to_numeric(df['yt'], errors='coerce')

bounds = {
    'height':     (100, 250),
    'weight':     (20, 200),
    'fb_friends': (0, 1000000),
    'yt':         (0, 10000),
}
for df in [train, test]:
    for col, (lo, hi) in bounds.items():
        df.loc[~df[col].between(lo, hi), col] = np.nan

# 3. 衍生特徵（只保留 BMI，移除與其共線的 height_weight_ratio）
for df in [train, test]:
    height_m = df['height'] / 100
    df['bmi'] = df['weight'] / (height_m ** 2)
    df['log_fb_friends'] = np.log1p(df['fb_friends'])

# 4. self_intro Embedding（BGE-large 1024 維，有快取就直接載入）
if os.path.exists('train_intro_emb.npy') and os.path.exists('test_intro_emb.npy'):
    print("載入已存在的 embedding 快取...")
    train_emb = np.load('train_intro_emb.npy')
    test_emb  = np.load('test_intro_emb.npy')
else:
    print("生成 self_intro embedding...")
    model_st = SentenceTransformer('baai/bge-large-en-v1.5')

    def get_embeddings(texts, fill_missing=''):
        filled = [str(t) if pd.notna(t) and str(t).strip() != ''
                  else fill_missing for t in texts]
        return model_st.encode(filled, show_progress_bar=True)

    train_emb = get_embeddings(train['self_intro'])
    test_emb  = get_embeddings(test['self_intro'])

    np.save('train_intro_emb.npy', train_emb)
    np.save('test_intro_emb.npy',  test_emb)

print(f"Embedding shape: {train_emb.shape}")

# 5. PCA 降維 + 合併特徵
N_COMPONENTS = 10
pca = PCA(n_components=N_COMPONENTS, random_state=42)
train_emb_pca = pca.fit_transform(train_emb)
test_emb_pca  = pca.transform(test_emb)
print(f"PCA 解釋變異量（{N_COMPONENTS} 維）：{pca.explained_variance_ratio_.sum():.1%}")

train_intro_missing = train['self_intro'].isnull().astype(int).values.reshape(-1, 1)
test_intro_missing  = test['self_intro'].isnull().astype(int).values.reshape(-1, 1)

base_features = ['height', 'weight', 'fb_friends', 'bmi', 'log_fb_friends']
feature_names = base_features + [f'pca_{i}' for i in range(N_COMPONENTS)] + ['intro_missing']

X_base_train = train[base_features].values
X_base_test  = test[base_features].values

X_train_full = np.hstack([X_base_train, train_emb_pca, train_intro_missing])
X_test_full  = np.hstack([X_base_test,  test_emb_pca,  test_intro_missing])

y_train = train['gender'].values

print(f"最終特徵維度：{X_train_full.shape[1]}")

# 6. Label encoding + 建立 pipeline（MissForest → SMOTE → Stacking）
label_map = {v: i for i, v in enumerate(sorted(np.unique(y_train)))}
y_encoded = np.array([label_map[v] for v in y_train])
inv_label_map = {v: k for k, v in label_map.items()}

print(f"Label mapping：{label_map}")
print(f"原始類別分布：{dict(zip(*np.unique(y_train, return_counts=True)))}\n")

stacking_model = StackingClassifier(
    estimators=[
        ('lgb', lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            min_child_samples=10, random_state=42, verbose=-1)),
        ('xgb', XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            min_child_samples=10, random_state=42, verbosity=0,
            use_label_encoder=False, eval_metric='logloss')),
        ('rf', RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=5,
            random_state=42, n_jobs=-1)),
    ],
    final_estimator=LogisticRegression(random_state=42, max_iter=1000),
    cv=5,
    passthrough=False
)

smote = SMOTE(random_state=42)
pipe = ImbPipeline([
    ('imputer', IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
        ),
        max_iter=10, random_state=42
    )),
    ('smote', smote),
    ('model', stacking_model)
])

# 7. Permutation Importance 特徵篩選（用 LightGBM 快速篩選，移除噪聲特徵）
print("=== Permutation Importance 分析 ===")

missforest_imputer = IterativeImputer(
    estimator=RandomForestRegressor(
        n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
    ),
    max_iter=10, random_state=42
)
X_train_imputed = missforest_imputer.fit_transform(X_train_full)

model_temp = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    min_child_samples=10, random_state=42, verbose=-1
)

# 每折只對訓練集做 SMOTE，驗證集保持原始分布
smote_temp = SMOTE(random_state=42)
cv_pi = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
perm_scores = []
for tr_idx, val_idx in cv_pi.split(X_train_imputed, y_encoded):
    X_tr, y_tr = X_train_imputed[tr_idx], y_encoded[tr_idx]
    X_val, y_val = X_train_imputed[val_idx], y_encoded[val_idx]

    X_tr_sm, y_tr_sm = smote_temp.fit_resample(X_tr, y_tr)
    model_temp.fit(X_tr_sm, y_tr_sm)

    result = permutation_importance(
        model_temp, X_val, y_val,
        n_repeats=10, random_state=42, scoring='accuracy'
    )
    perm_scores.append(result.importances_mean)

perm_importance_mean = np.mean(perm_scores, axis=0)

print("\n特徵 Permutation Importance 排名：")
sorted_idx = np.argsort(perm_importance_mean)[::-1]
for i in sorted_idx:
    print(f"  {feature_names[i]:25s} : {perm_importance_mean[i]:+.4f}")

keep_mask = perm_importance_mean > 0
removed_features = [feature_names[i] for i in range(len(feature_names)) if not keep_mask[i]]
kept_features = [feature_names[i] for i in range(len(feature_names)) if keep_mask[i]]

if removed_features:
    print(f"\n移除噪聲特徵（importance <= 0）：{removed_features}")
else:
    print("\n沒有需要移除的噪聲特徵")

print(f"保留特徵（{len(kept_features)}）：{kept_features}")

X_train = X_train_full[:, keep_mask]
X_test  = X_test_full[:, keep_mask]

# 8. 5-Fold Stratified CV
N_FOLDS = 5
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

print(f"\n=== {N_FOLDS}-Fold Stratified CV 結果（篩選後 {X_train.shape[1]} 特徵）===")

all_val_preds = np.zeros(len(y_encoded), dtype=int)
fold_accs = []
for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_encoded)):
    pipe.fit(X_train[tr_idx], y_encoded[tr_idx])
    preds = pipe.predict(X_train[val_idx])
    all_val_preds[val_idx] = preds
    acc = accuracy_score(y_encoded[val_idx], preds)
    fold_accs.append(acc)
    print(f"  Fold {fold+1}: accuracy = {acc:.4f}")

print(f"\nCV Mean accuracy : {np.mean(fold_accs):.4f}")
print(f"CV Std  accuracy : {np.std(fold_accs):.4f}")

oof_labels = [inv_label_map[p] for p in all_val_preds]
true_labels = y_train.tolist()
print(classification_report(true_labels, oof_labels))

# 9. 完整訓練 + 輸出 submission
pipe.fit(X_train, y_encoded)
test_preds_encoded = pipe.predict(X_test)
test_preds = [inv_label_map[p] for p in test_preds_encoded]

submission = pd.DataFrame({
    'id':     test['id'],
    'gender': test_preds
})

out_file = '../submission/submission_PI_bge_v2.csv'
submission.to_csv(out_file, index=False)
print(f"\n{out_file} 已輸出（{len(submission)} 筆）")
print(f"預測分布：")
print(pd.Series(test_preds).value_counts(normalize=True).round(3))
