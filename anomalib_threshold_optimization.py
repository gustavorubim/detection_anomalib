import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, confusion_matrix, accuracy_score, recall_score, precision_score
)

# === Load data ===
file_path = "./inference_analysis/anomaly_detection_results.csv"
df = pd.read_csv(file_path)

# === Map ground truth labels to binary ===
df['Ground_truth_bin'] = df['Ground_truth'].map({'good': 0, 'Normal': 0, 'Anomalous': 1})

# === Identify model score columns ===
score_columns = [col for col in df.columns if col.endswith('_score')]

# === Tune thresholds using cross-validation ===
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
thresholds = np.linspace(0, 1, 101)
results = {}

for score_col in score_columns:
    X = df[score_col].values
    y = df['Ground_truth_bin'].values
    best_thresh = 0.5
    best_f1 = 0

    for thresh in thresholds:
        f1s = []
        for train_idx, val_idx in cv.split(X, y):
            preds = (X[val_idx] > thresh).astype(int)
            f1 = f1_score(y[val_idx], preds)
            f1s.append(f1)
        avg_f1 = np.mean(f1s)
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_thresh = thresh

    results[score_col] = {'best_threshold': best_thresh, 'cv_f1': best_f1}

# === Apply optimal thresholds to generate new labels ===
for score_col, values in results.items():
    threshold = values['best_threshold']
    new_col_name = f"{score_col}_threshold_{threshold:.2f}"
    df[new_col_name] = df[score_col].apply(lambda x: 'Anomalous' if x > threshold else 'good')

# === Save updated DataFrame ===
output_file = "./inference_analysis/anomaly_detection_thresholded.csv"
df.to_csv(output_file, index=False)

# === Define metric computation ===
def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = recall
    return {
        'accuracy': accuracy,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'recall': recall,
        'precision': precision
    }

# === Map string labels to binary ===
def label_to_binary(series):
    return series.map({'good': 0, 'Normal': 0, 'Anomalous': 1}).fillna(-1).astype(int)

# === Compute and print metrics summary ===
metrics_summary = []

for score_col in score_columns:
    label_col = score_col.replace("_score", "_label")
    threshold = results[score_col]['best_threshold']
    enhanced_col = f"{score_col}_threshold_{threshold:.2f}"

    y_true = df['Ground_truth_bin'].values
    y_orig = label_to_binary(df[label_col])
    y_enhanced = label_to_binary(df[enhanced_col])

    if (y_orig == -1).any() or (y_enhanced == -1).any():
        continue

    orig_metrics = compute_metrics(y_true, y_orig)
    enhanced_metrics = compute_metrics(y_true, y_enhanced)

    metrics_summary.append({
        'model': score_col.replace('_score', ''),
        'type': 'original',
        **orig_metrics
    })
    metrics_summary.append({
        'model': score_col.replace('_score', ''),
        'type': 'threshold_optimized',
        **enhanced_metrics
    })

# === Display metrics table ===
metrics_df = pd.DataFrame(metrics_summary)
pd.set_option('display.float_format', '{:.4f}'.format)
print("\n=== Model Metrics Comparison ===")
print(metrics_df.to_string(index=False))

print(f"\nUpdated CSV saved to: {output_file}")
