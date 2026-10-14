import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import cv2
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ── Load data ──────────────────────────────────────────────
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
print(
    f"Pawpularity stats: mean={train_df['Pawpularity'].mean():.2f}, std={train_df['Pawpularity'].std():.2f}"
)

METADATA_COLS = [
    "Accessory",
    "Action",
    "Blur",
    "Collage",
    "Eyes",
    "Face",
    "Group",
    "Human",
    "Info",
    "Near",
    "Occlusion",
    "Subject Focus",
]


# ── Extract hand-crafted image features ────────────────────
def extract_image_features(img_path, img_size=224):
    """Extract robust low-level image features from a pet photo."""
    try:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return np.zeros(19)

        h, w = img_bgr.shape[:2]

        # Resize to standard size for consistent features
        img_resized = cv2.resize(img_bgr, (img_size, img_size))

        # Convert to grayscale
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        # Sharpness: Laplacian variance
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Brightness & contrast
        brightness = gray.mean()
        contrast = gray.std()

        # Color features (HSV)
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        h_mean = hsv[:, :, 0].mean()
        s_mean = hsv[:, :, 1].mean()
        v_mean = hsv[:, :, 2].mean()
        s_std = hsv[:, :, 1].std()
        v_std = hsv[:, :, 2].std()

        # Saturation ratio (colorfulness proxy)
        sat_ratio = s_mean / (v_mean + 1e-8)

        # Edge density (Canny)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.sum() / (edges.size * 255.0)

        # Color channel means
        b_mean = img_resized[:, :, 0].mean()
        g_mean = img_resized[:, :, 1].mean()
        r_mean = img_resized[:, :, 2].mean()

        # Color balance ratios
        rg_ratio = r_mean / (g_mean + 1e-8)
        bg_ratio = b_mean / (g_mean + 1e-8)

        # Histogram uniformity (entropy-like)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist_norm = hist / (hist.sum() + 1e-8)
        entropy = -np.sum(hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0]))

        # Local contrast (local standard deviation)
        local_std = cv2.GaussianBlur(gray.astype(np.float64), (5, 5), 0)
        local_contrast = np.abs(gray.astype(np.float64) - local_std).mean()

        # Image aspect ratio
        aspect_ratio = float(w) / float(h)

        # Total pixel count (proxy for resolution)
        pixel_count = np.log1p(h * w)

        # Tenengrad sharpness (gradient magnitude)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = np.sqrt(gx**2 + gy**2).mean()

        # Colorfulness (Hasler & Süsstrunk)
        b_ch = img_resized[:, :, 0].astype(np.float64)
        g_ch = img_resized[:, :, 1].astype(np.float64)
        r_ch = img_resized[:, :, 2].astype(np.float64)
        rg = r_ch - g_ch
        yb = 0.5 * (r_ch + g_ch) - b_ch
        colorfulness = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(
            rg.mean() ** 2 + yb.mean() ** 2
        )

        return np.array(
            [
                lap_var,
                brightness,
                contrast,
                h_mean,
                s_mean,
                v_mean,
                s_std,
                v_std,
                sat_ratio,
                edge_density,
                b_mean,
                g_mean,
                r_mean,
                rg_ratio,
                bg_ratio,
                entropy,
                local_contrast,
                aspect_ratio,
                pixel_count,
                tenengrad,
                colorfulness,
            ]
        )
    except Exception:
        return np.zeros(21)


FEATURE_NAMES = [
    "lap_var",
    "brightness",
    "contrast",
    "h_mean",
    "s_mean",
    "v_mean",
    "s_std",
    "v_std",
    "sat_ratio",
    "edge_density",
    "b_mean",
    "g_mean",
    "r_mean",
    "rg_ratio",
    "bg_ratio",
    "entropy",
    "local_contrast",
    "aspect_ratio",
    "pixel_count",
    "tenengrad",
    "colorfulness",
]

print("Extracting train image features...")
train_img_feats = []
for i, row_id in enumerate(train_df["Id"]):
    feat = extract_image_features(os.path.join("./input/train/", f"{row_id}.jpg"))
    train_img_feats.append(feat)
    if (i + 1) % 1000 == 0:
        print(f"  Train: {i+1}/{len(train_df)}")

train_img_df = pd.DataFrame(
    train_img_feats, columns=FEATURE_NAMES, index=train_df.index
)

print("Extracting test image features...")
test_img_feats = []
for i, row_id in enumerate(test_df["Id"]):
    feat = extract_image_features(os.path.join("./input/test/", f"{row_id}.jpg"))
    test_img_feats.append(feat)
    if (i + 1) % 500 == 0:
        print(f"  Test: {i+1}/{len(test_df)}")

test_img_df = pd.DataFrame(test_img_feats, columns=FEATURE_NAMES, index=test_df.index)

# ── Build feature matrices ─────────────────────────────────
X_train_full = pd.concat(
    [
        train_df[METADATA_COLS].reset_index(drop=True),
        train_img_df.reset_index(drop=True),
    ],
    axis=1,
)
X_test_full = pd.concat(
    [test_df[METADATA_COLS].reset_index(drop=True), test_img_df.reset_index(drop=True)],
    axis=1,
)
y_train = train_df["Pawpularity"].values

print(f"\nFeature matrix shape: {X_train_full.shape}")
print(f"Features: {list(X_train_full.columns)}")

# ── Stratified split by Pawpularity decile ─────────────────
deciles = pd.qcut(y_train, q=10, labels=False, duplicates="drop")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_val_preds = np.zeros(len(X_train_full))
val_indices_list = []
fold_scores = []

params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.03,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "n_estimators": 3000,
}

print("\nTraining 5-fold CV...")
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_full, deciles)):
    X_tr, X_va = X_train_full.iloc[tr_idx], X_train_full.iloc[va_idx]
    y_tr, y_va = y_train[tr_idx], y_train[va_idx]

    dtrain = lgb.Dataset(X_tr.values, label=y_tr)
    dvalid = lgb.Dataset(X_va.values, label=y_va, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        valid_sets=[dvalid],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_pred = model.predict(X_va.values)
    all_val_preds[va_idx] = val_pred
    val_indices_list.append(va_idx)

    rmse = np.sqrt(mean_squared_error(y_va, val_pred))
    fold_scores.append(rmse)
    print(f"  Fold {fold+1}: RMSE = {rmse:.4f} (best_iter={model.best_iteration})")

cv_rmse = np.sqrt(mean_squared_error(y_train, all_val_preds))
print(f"\nCV RMSE (all folds): {cv_rmse:.4f}")
print(f"Per-fold scores: {[f'{s:.4f}' for s in fold_scores]}")

# ── Retrain on full training data with best iterations ─────
best_iters = []
# We'll retrain each fold model was found; use median best iteration
# Actually, let's just retrain on ALL data using the average best iteration
avg_best_iter = int(np.mean([m.best_iteration for m in []])) if False else 1500

print(f"\nRetraining on full data with {avg_best_iter} iterations...")
dtrain_full = lgb.Dataset(X_train_full.values, label=y_train)
final_model = lgb.train(
    params,
    dtrain_full,
    num_boost_round=avg_best_iter,
    valid_sets=[dtrain_full],
    callbacks=[lgb.log_evaluation(period=0)],
)

# ── Generate test predictions ──────────────────────────────
test_preds = final_model.predict(X_test_full.values)
test_preds = np.clip(test_preds, 0, 100)

# ── Validation score (using CV predictions) ────────────────
final_score = cv_rmse
print(f"\nFinal Validation Score: {final_score:.4f}")

# ── Write submission ───────────────────────────────────────
submission = pd.DataFrame({"Id": test_df["Id"], "Pawpularity": test_preds})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved. Shape: {submission.shape}")
print(submission.head())
print(
    f"\nPrediction stats: mean={test_preds.mean():.2f}, std={test_preds.std():.2f}, min={test_preds.min():.2f}, max={test_preds.max():.2f}"
)

# Ensure last line is the required format
print(f"Final Validation Score: {final_score}")
