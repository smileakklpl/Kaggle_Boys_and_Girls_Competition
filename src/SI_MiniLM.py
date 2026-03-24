# =============================================================
# PCA v2 — MiniLM (384d) + Grid Search (PCA 維度 x alpha)
# 改動：MiniLM 取代 bge-large、移除 intro_missing
# =============================================================

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import lightgbm as lgb

# ----------------------------------------------------------
# 1. 載入資料 + 清洗
# ----------------------------------------------------------
train = pd.read_csv('../data/boy_or_girl_2025_train_missingValue.csv')
test  = pd.read_csv('../data/boy_or_girl_2025_test_no_ans_missingValue.csv')

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
# 2. self_intro Embedding — MiniLM (384d)
# ----------------------------------------------------------
TRAIN_EMB_PATH = 'train_intro_emb_minilm.npy'
TEST_EMB_PATH  = 'test_intro_emb_minilm.npy'

if os.path.exists(TRAIN_EMB_PATH) and os.path.exists(TEST_EMB_PATH):
    print("載入已存在的 MiniLM embedding 快取...")
    train_emb = np.load(TRAIN_EMB_PATH)
    test_emb  = np.load(TEST_EMB_PATH)
else:
    print("生成 MiniLM embedding（all-MiniLM-L6-v2, 384d）...")
    model_st = SentenceTransformer('all-MiniLM-L6-v2')

    def get_embeddings(texts):
        filled = [str(t) if pd.notna(t) and str(t).strip() != '' else '' for t in texts]
        return model_st.encode(filled, show_progress_bar=True)

    train_emb = get_embeddings(train['self_intro'])
    test_emb  = get_embeddings(test['self_intro'])
    np.save(TRAIN_EMB_PATH, train_emb)
    np.save(TEST_EMB_PATH, test_emb)

print(f"Embedding shape: {train_emb.shape}")

# ----------------------------------------------------------
# 3. 模型設定
# ----------------------------------------------------------
base_features = ['height', 'weight', 'fb_friends']
y_train = train['gender'].values

label_map = {v: i for i, v in enumerate(sorted(np.unique(y_train)))}
y_encoded = np.array([label_map[v] for v in y_train])
inv_label_map = {v: k for k, v in label_map.items()}

n_total = len(y_encoded)
class_weight_dict = {
    label_map[cls]: n_total / (len(label_map) * count)
    for cls, count in zip(*np.unique(y_train, return_counts=True))
}
print(f"Class weights：{class_weight_dict}\n")

# ----------------------------------------------------------
# 4. Grid Search：PCA 維度 x alpha
# ----------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for n_comp in [5, 10, 15, 20]:
    pca_tmp = PCA(n_components=n_comp, random_state=42)
    train_pca = pca_tmp.fit_transform(train_emb)
    explained = pca_tmp.explained_variance_ratio_.sum()

    for alpha in [0.3, 0.5, 0.7, 1.0, 1.5]:
        X_train = np.hstack([train[base_features].values, train_pca * alpha])

        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', lgb.LGBMClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=4,
                min_child_samples=10, class_weight=class_weight_dict,
                random_state=42, verbose=-1
            ))
        ])

        oof = cross_val_predict(pipe, X_train, y_encoded, cv=cv)
        oof_labels = [inv_label_map[p] for p in oof]
        report = classification_report(y_train, oof_labels, output_dict=True)

        results.append({
            'n_comp': n_comp, 'alpha': alpha, 'explained': explained,
            'accuracy':  round(report['accuracy'], 4),
            'g2_recall': round(report['2']['recall'], 4),
            'g2_f1':     round(report['2']['f1-score'], 4),
        })

results_df = pd.DataFrame(results)

print("=== Grid Search 結果（按 g2_recall 排序）===")
print(results_df.sort_values('g2_recall', ascending=False).to_string(index=False))

best = results_df.loc[results_df['g2_recall'].idxmax()]
print(f"\n最佳：n_comp={int(best['n_comp'])}, alpha={best['alpha']}"
      f" → g2_recall={best['g2_recall']}, acc={best['accuracy']}")

# ----------------------------------------------------------
# 5. 用最佳參數訓練 + CV 報告 + 輸出 submission
# ----------------------------------------------------------
best_n = int(best['n_comp'])
best_alpha = best['alpha']

pca = PCA(n_components=best_n, random_state=42)
train_emb_pca = pca.fit_transform(train_emb)
test_emb_pca  = pca.transform(test_emb)

X_train = np.hstack([train[base_features].values, train_emb_pca * best_alpha])
X_test  = np.hstack([test[base_features].values,  test_emb_pca  * best_alpha])

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model', lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        min_child_samples=10, class_weight=class_weight_dict,
        random_state=42, verbose=-1
    ))
])

print(f"\n=== 最佳參數 CV 報告（n_comp={best_n}, alpha={best_alpha}）===")
oof = cross_val_predict(pipe, X_train, y_encoded, cv=cv)
print(classification_report(y_train, [inv_label_map[p] for p in oof]))

fold_accs = []
for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_encoded)):
    pipe.fit(X_train[tr_idx], y_encoded[tr_idx])
    preds = pipe.predict(X_train[val_idx])
    acc = accuracy_score(y_encoded[val_idx], preds)
    fold_accs.append(acc)
    print(f"  Fold {fold+1}: accuracy = {acc:.4f}")

print(f"\nCV Mean accuracy : {np.mean(fold_accs):.4f}")
print(f"CV Std  accuracy : {np.std(fold_accs):.4f}")

pipe.fit(X_train, y_encoded)
test_preds = [inv_label_map[p] for p in pipe.predict(X_test)]

submission = pd.DataFrame({'id': test['id'], 'gender': test_preds})
submission.to_csv('submission_pca_v2.csv', index=False)
print(f"submission_pca_v2.csv 已輸出（{len(submission)} 筆）")
print(f"預測分布：")
print(pd.Series(test_preds).value_counts(normalize=True).round(3))
