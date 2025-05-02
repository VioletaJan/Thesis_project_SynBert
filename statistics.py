import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score

# Confidence Interval function for Accuracy, Precision, Recall
def compute_confidence_interval(p, n):
    margin = 1.96 * np.sqrt(p * (1 - p) / n)
    lower = round(p - margin, 4)
    upper = round(p + margin, 4)
    return f"({lower}, {upper})"

def evaluate_dataset(name, df):
    y_true = df['Actual']
    y_pred = df['Predicted']
    n = len(df)

    acc = round(accuracy_score(y_true, y_pred), 4)
    pre = round(precision_score(y_true, y_pred), 4)
    rec = round(recall_score(y_true, y_pred), 4)

    return {
        'Dataset': name,
        'Accuracy': acc,
        'CI Accuracy': compute_confidence_interval(acc, n),
        'Precision': pre,
        'CI Precision': compute_confidence_interval(pre, n),
        'Recall': rec,
        'CI Recall': compute_confidence_interval(rec, n),
    }

# Confidence Interval function to calculate bootstrap F1-score 
from sklearn.metrics import f1_score
from sklearn.utils import resample

def bootstrap_f1_confidence_interval(y_true, y_pred, n_bootstrap=1000, random_state=None):
    rng = np.random.default_rng(random_state)
    f1_scores = []

    data = list(zip(y_true, y_pred))

    for _ in range(n_bootstrap):
        sample = resample(data, replace=True, random_state=rng.integers(1e6))
        y_true_sample, y_pred_sample = zip(*sample)
        f1 = f1_score(y_true_sample, y_pred_sample, zero_division=0)
        f1_scores.append(f1)

    ci_lower = np.percentile(f1_scores, 2.5)
    ci_upper = np.percentile(f1_scores, 97.5)

    return ci_lower, ci_upper


# Function to calculate FAR for each prediction set
from sklearn.metrics import confusion_matrix

far_results = []

for name, df in predictions.items():
    y_true = df['Actual'].values
    y_pred = df['Predicted'].values  

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    if (fp + tn) == 0:
        far = 0.0
    else:
        far = fp / (fp + tn)

    far_results.append({
        'Model': name,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TP': tp,
        'FAR': round(far, 4)
    })


# Function for McNemar's test
from statsmodels.stats.contingency_tables import mcnemar

def run_mcnemar_test(actual, pred1, pred2, model1, model2):
    n_01 = ((pred1 != actual) & (pred2 == actual)).sum()
    n_10 = ((pred1 == actual) & (pred2 != actual)).sum()
    result = mcnemar([[0, n_10], [n_01, 0]], exact=False, correction=True)

    return {
        'Comparison': f'{model1} vs {model2}',
        'n_01': n_01,
        'n_10': n_10,
        'Statistic': round(result.statistic, 4),
        'p-value': f"{result.pvalue:.2e}".replace('e-', 'E−'),
        'Significant (p < 0.05)': 'Yes' if result.pvalue < 0.05 else 'No'
    }


