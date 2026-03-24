# =============================================================
# Submission 3 — LightGBM + self_intro embedding
# 特徵：height, weight, fb_friends + self_intro PCA embedding
# 模型：LightGBM + class_weight
# 最佳成績：acc = 0.8497
# =============================================================

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import os

# ----------------------------------------------------------
# 1. 載入資料
# ----------------------------------------------------------
train = pd.read_csv('../data/boy_or_girl_2025_train_missingValue.csv')
test  = pd.read_csv('../data/boy_or_girl_2025_test_no_ans_missingValue.csv')

# ----------------------------------------------------------
# 2. 資料清洗
# ----------------------------------------------------------
for df in [train, test]:
    df['yt'] = pd.to_numeric(df['yt'], errors='coerce')

bounds = {
    'height':     (100, 250),
    'weight':     (20, 200),
    'fb_friends': (0, 10000),
    'yt':         (0, 100),
}
for df in [train, test]:
    for col, (lo, hi) in bounds.items():
        df.loc[~df[col].between(lo, hi), col] = np.nan

# ----------------------------------------------------------
# 3. self_intro Embedding（有快取就直接載入）
# ----------------------------------------------------------
if os.path.exists('train_intro_emb.npy') and os.path.exists('test_intro_emb.npy'):
    print("載入已存在的 embedding 快取...")
    train_emb = np.load('train_intro_emb.npy')
    test_emb  = np.load('test_intro_emb.npy')
else:
    print("生成 self_intro embedding...")
    model_st = SentenceTransformer('BAAI/bge-large-en-v1.5')

    def get_embeddings(texts, fill_missing=''):
        filled = [str(t) if pd.notna(t) and str(t).strip() != ''
                  else fill_missing for t in texts]
        return model_st.encode(filled, show_progress_bar=True)

    train_emb = get_embeddings(train['self_intro'])
    test_emb  = get_embeddings(test['self_intro'])

    np.save('train_intro_emb.npy', train_emb)
    np.save('test_intro_emb.npy',  test_emb)

print(f"Embedding shape: {train_emb.shape}")

# ----------------------------------------------------------
# 4. PCA 降維 + 合併特徵
# ----------------------------------------------------------
N_COMPONENTS = 10
pca = PCA(n_components=N_COMPONENTS, random_state=42)
train_emb_pca = pca.fit_transform(train_emb)
test_emb_pca  = pca.transform(test_emb)
print(f"PCA 解釋變異量：{pca.explained_variance_ratio_.sum():.1%}")

# is_missing indicator
train_intro_missing = train['self_intro'].isnull().astype(int).values.reshape(-1, 1)
test_intro_missing  = test['self_intro'].isnull().astype(int).values.reshape(-1, 1)

# 合併：base 數值特徵 + PCA embedding + intro_missing
base_features = ['height', 'weight', 'fb_friends']
X_base_train = train[base_features].values
X_base_test  = test[base_features].values

X_train = np.hstack([X_base_train, train_emb_pca, train_intro_missing])
X_test  = np.hstack([X_base_test,  test_emb_pca,  test_intro_missing])

y_train = train['gender'].values

print(f"最終特徵維度：{X_train.shape[1]}")
# base(3) + PCA(10) + missing(1) = 14

# ----------------------------------------------------------
# 5. LightGBM + Median 補值（base 特徵有 NaN）
# ----------------------------------------------------------
label_map = {v: i for i, v in enumerate(sorted(np.unique(y_train)))}
y_encoded = np.array([label_map[v] for v in y_train])
inv_label_map = {v: k for k, v in label_map.items()}

n_total = len(y_encoded)
class_weight_dict = {
    label_map[cls]: n_total / (len(label_map) * count)
    for cls, count in zip(*np.unique(y_train, return_counts=True))
}
print(f"Label mapping：{label_map}")
print(f"Class weights：{class_weight_dict}\n")

model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    min_child_samples=10,
    class_weight=class_weight_dict,
    random_state=42,
    verbose=-1
)

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model', model)
])

# ----------------------------------------------------------
# 6. Cross-Validation
# ----------------------------------------------------------
N_FOLDS = 5
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

print(f"=== {N_FOLDS}-Fold Stratified CV 結果 ===")
oof_preds = cross_val_predict(pipe, X_train, y_encoded, cv=cv, method='predict')

oof_labels = [inv_label_map[p] for p in oof_preds]
true_labels = y_train.tolist()
print(classification_report(true_labels, oof_labels))

fold_accs = []
for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_encoded)):
    pipe.fit(X_train[tr_idx], y_encoded[tr_idx])
    preds = pipe.predict(X_train[val_idx])
    acc = accuracy_score(y_encoded[val_idx], preds)
    fold_accs.append(acc)
    print(f"  Fold {fold+1}: accuracy = {acc:.4f}")

print(f"\nCV Mean accuracy : {np.mean(fold_accs):.4f}")
print(f"CV Std  accuracy : {np.std(fold_accs):.4f}")

# ----------------------------------------------------------
# 7. 完整訓練 + 輸出 submission
# ----------------------------------------------------------
pipe.fit(X_train, y_encoded)
test_preds_encoded = pipe.predict(X_test)
test_preds = [inv_label_map[p] for p in test_preds_encoded]

submission = pd.DataFrame({
    'id':     test['id'],
    'gender': test_preds
})

submission.to_csv('submission4.csv', index=False)
print(f"\nsubmission4.csv 已輸出（{len(submission)} 筆）")
print(f"預測分布：")
print(pd.Series(test_preds).value_counts(normalize=True).round(3))
