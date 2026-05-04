# Kaggle Boys and Girls Competition

本專案為 Kaggle 競賽作業，目標是根據受訪者的身體測量數據、行為習慣與自我介紹文字，預測其性別標籤（Gender = 1 或 2）。最終版本以 `ensemble_ultimate_v4.ipynb` 為核心實作，整合了 NLP 語意嵌入（BGE-Large）、多重防漏交叉驗證、Stacking 集成學習與三檔位 Blending 融合策略，在 Stratified 10-Fold CV 上達到 **Accuracy 87.92%、ROC-AUC 0.9215**。

---

## 目錄

1. [專案結構](#專案結構)
2. [資料集說明](#資料集說明)
3. [特徵工程與資料前處理](#特徵工程與資料前處理)
4. [模型架構](#模型架構)
5. [效能評估結果](#效能評估結果)
6. [安裝與執行](#安裝與執行)
7. [套件依賴](#套件依賴)

---

## 專案結構

```text
Kaggle_Boys_and_Girls_Competition/
├── data/                                            # 原始資料集
│   ├── boy or girl 2025 train_missingValue.csv      # 訓練集（含部分缺失值）
│   ├── boy or girl 2025 test no ans_missingValue.csv # 測試集（無標籤）
│   └── Boy_or_girl_test_sandbox_sample_submission.csv # 繳交格式範例
├── src/
│   └── ensemble_ultimate_v4.ipynb                  # 最終版本主程式
├── results/
│   ├── feature_vote_counts.png                     # 特徵投票重要性圖
│   └── submission_blending_ultimate.csv            # 最終預測繳交檔案
├── docs/
│   └── Research Report.pdf                        # 研究報告（完整版）
├── requirements.txt                                # 相依套件版本列表
├── pyproject.toml                                  # 專案設定
└── README.md
```

---

## 資料集說明

訓練集共 11 個欄位，部分欄位含有缺失值或異常值（如科學記號的極端數字、Excel 錯誤碼 `#NUM!`）：

| 欄位名稱 | 類型 | 說明 |
|---|---|---|
| `id` | 整數 | 樣本識別碼 |
| `gender` | 整數（1 / 2） | 目標標籤（性別） |
| `star_sign` | 類別 | 星座（中文，如「處女座」） |
| `phone_os` | 類別 | 手機作業系統（Apple / Android / 其他） |
| `height` | 數值 | 身高（公分） |
| `weight` | 數值 | 體重（公斤） |
| `sleepiness` | 數值 | 每日睡眠時數 |
| `iq` | 數值 | IQ 數值 |
| `fb_friends` | 數值 | Facebook 好友數 |
| `yt` | 數值 | YouTube 使用時數（含大量異常值） |
| `self_intro` | 文字 | 自我介紹（英文短文） |

---

## 特徵工程與資料前處理

整個 Pipeline 設計核心原則是**嚴格防止資料洩漏 (Data Leakage)**，所有 Encoding、補值與降維操作均在 Cross-Validation 的 Fold 內部進行。

### 1. 資料清洗與異常值過濾

針對各數值特徵設定合理物理邊界，超出範圍者設為 `NaN`：

| 特徵 | 下限 | 上限 | 說明 |
|---|---|---|---|
| `height` | 100 cm | 250 cm | 排除負值與科學記號溢位 |
| `weight` | 30 kg | 200 kg | 排除負值與極端異常值 |
| `iq` | 50 | 200 | 合理人類 IQ 範圍 |
| `sleepiness` | 0 hr | 24 hr | 每日最多 24 小時 |
| `fb_friends` | 0 | 5000 | 排除負值 |
| `yt` | 0 | 99th 百分位數 | 動態上限，避免 Excel 錯誤碼污染 |

`phone_os` 欄位透過模糊字串比對統一化（含 `apple`/`ios` → `Apple`，含 `android` → `Android`，其餘 → `Other`）。

### 2. 缺失值指示器

對 `height`、`weight`、`yt` 三個高缺失率欄位建立二元遮罩特徵（`xxx_is_missing`），保留「該資料是否缺失」本身就可能含有的性別信號。

### 3. NLP 語意嵌入（BGE-Large）

使用 `BAAI/bge-large-en-v1.5` 模型將 `self_intro` 文字欄位轉換為高維語意向量，再透過 **PCA 降至 5 維**（防洩漏版：PCA 在每個 Fold 內部重新 `fit`）。

### 4. 類別特徵編碼（Target Encoding）

`star_sign` 與 `phone_os` 使用 **Target Encoding（smoothing=20）**，同樣在 Fold 內部 `fit_transform` 以防止洩漏。

### 5. 缺失值補值（Random Forest Imputer）

使用 `IterativeImputer` 搭配 `RandomForestRegressor`（50棵樹，`max_iter=10`）進行多重補值，透過消融實驗（Ablation Study）與 Simple/KNN/LightGBM Imputer 對比後選用。

### 6. 衍生特徵工程

| 衍生特徵 | 計算公式 | 說明 |
|---|---|---|
| `BMI` | weight / (height/100)² | 身體質量指數 |
| `height_weight_ratio` | height / weight | 身高體重線性比 |
| `yt_log` | log1p(yt) | 對右偏分布取 Log |
| `fb_friends_log` | log1p(fb_friends) | 對右偏分布取 Log |

原始 `yt` 與 `fb_friends` 在建立 Log 特徵後移除。

### 7. 特徵篩選（多輪投票制 Permutation Importance）

使用 **5 個不同隨機種子** 各跑一輪 5-Fold 的 Permutation Importance（基礎模型：LightGBM），對每個特徵統計在幾輪中被判定為「有效特徵（PI > 0）」。只要在 **≥ 3 輪中獲得認可**，就保留此特徵；否則視為噪聲移除。最終保留 16 個核心特徵。

---

## 模型架構

### 三檔位 Stacking + Blending 融合

最終模型由三組不同「性格」的 Stacking Ensemble 組合而成，透過對預測機率平均（Blending）來提升穩定性與泛化能力：

| 配置 | XGBoost | LightGBM | Random Forest |
|---|---|---|---|
| **保守 (Conservative)** | depth=3, lr=0.05, n=150, λ=5.0 | depth=3, leaves=15, lr=0.05, n=150 | depth=4, n=150, min_leaf=4 |
| **中庸 (Moderate)** | depth=5, lr=0.03, n=200, λ=1.0 | depth=5, leaves=31, lr=0.03, n=200 | depth=6, n=200, min_leaf=2 |
| **激進 (Aggressive)** | depth=7, lr=0.01, n=300, λ=0.1 | depth=7, leaves=63, lr=0.01, n=300 | depth=8, n=300, min_leaf=1 |

每個 Stacking 的 Meta-Learner 為 **Logistic Regression（class_weight='balanced'）**。

XGBoost 使用 `scale_pos_weight = count(class 0) / count(class 1)` 處理類別不平衡，LightGBM 與 RF 使用 `class_weight='balanced'`。

---

## 效能評估結果

採用 **Stratified 10-Fold Cross-Validation** 在訓練集上評估最終融合模型：

| 指標 | 均值 | 標準差 |
|---|---|---|
| Accuracy（準確率） | **0.8792** | ±0.0559 |
| ROC-AUC（排序力） | **0.9215** | ±0.0750 |
| F1-Score | **0.7841** | ±0.0908 |
| Log Loss（校準度） | **0.3868** | ±0.1163 |

### 補值策略消融實驗（10-Fold）

| 補值方法 | ROC-AUC | F1-Score |
|---|---|---|
| Simple Imputer (Median) | 0.9319 (±0.0607) | 0.7720 |
| KNN Imputer (k=5) | 0.9317 (±0.0639) | 0.7861 |
| LightGBM Imputer | 0.9308 (±0.0702) | 0.7943 |
| **Random Forest Imputer (最終選用)** | 0.9227 (±0.0727) | 0.7702 |

---

## 安裝與執行

### 環境需求

- Python 3.9+
- （建議）CUDA 相容 GPU（XGBoost 使用 `device='cuda'`，無 GPU 可將參數改為 `device='cpu'`）

### 安裝套件

```bash
pip install -r requirements.txt
```

### 執行流程

使用 Jupyter Notebook 或 VS Code 開啟 `src/ensemble_ultimate_v4.ipynb`，按順序執行各 Cell：

| Cell | 功能 |
|---|---|
| Cell 1 | 匯入套件、讀取資料 |
| Cell 2 | 資料清洗、邊界過濾、缺失值指示器 |
| Cell 3 | BGE-Large NLP 語意嵌入（首次執行需下載模型） |
| Cell 4 | 防漏預處理函數定義（Encoding、PCA、補值、特徵工程） |
| Cell 5 | 多輪投票制 Permutation Importance 特徵篩選並輸出圖表 |
| Cell 6 | 三檔位 Blending 10-Fold 評估 |
| Cell 7 | 全量訓練並輸出預測結果至 `results/` |

預測結果會自動儲存為 `results/submission_blending_ultimate.csv`，特徵重要性圖儲存為 `results/feature_vote_counts.png`。

---

## 套件依賴

| 套件 | 版本需求 | 用途 |
|---|---|---|
| `pandas` | ≥ 2.2.0 | 資料處理 |
| `numpy` | ≥ 1.26.0 | 數值計算 |
| `scikit-learn` | ≥ 1.4.0 | 模型訓練、評估、補值 |
| `xgboost` | ≥ 2.0.0 | 梯度提升樹（支援 CUDA） |
| `lightgbm` | ≥ 4.3.0 | 輕量梯度提升樹 |
| `category-encoders` | ≥ 2.6.0 | Target Encoding |
| `sentence-transformers` | ≥ 3.0.0 | BGE-Large 語意嵌入 |
| `optuna` | ≥ 3.6.0 | 超參數自動優化（備用） |
| `tf_keras` | ≥ 2.20.1 | sentence-transformers 後端依賴 |
| `matplotlib` | ≥ 3.8.0 | 資料視覺化 |
| `seaborn` | ≥ 0.13.2 | 統計圖表 |
| `ipykernel` | ≥ 6.29.0 | Jupyter Notebook 執行環境 |
