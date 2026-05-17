"""
train_models.py
================
Trains Naive Bayes, Decision Tree, and Random Forest on the disease dataset.
Saves models + label encoder + symptom list as pickle files.
Also generates evaluation charts saved to media/charts/.
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV  = os.path.join(BASE_DIR, 'Training.csv')
TEST_CSV   = os.path.join(BASE_DIR, 'Testing.csv')
MODELS_DIR = BASE_DIR  # save .pkl alongside data
CHARTS_DIR = os.path.join(BASE_DIR, '..', '..', 'media', 'charts')

os.makedirs(CHARTS_DIR, exist_ok=True)


def load_data():
    """Load and clean Training + Testing CSVs."""
    df_train = pd.read_csv(TRAIN_CSV)
    df_test  = pd.read_csv(TEST_CSV)

    # Strip extra whitespace from column names
    df_train.columns = df_train.columns.str.strip()
    df_test.columns  = df_test.columns.str.strip()

    # Fill NaN with 0
    df_train = df_train.fillna(0)
    df_test  = df_test.fillna(0)

    return df_train, df_test


def prepare_features(df_train, df_test):
    """Separate features and target, encode labels."""
    symptom_cols = [c for c in df_train.columns if c != 'prognosis']

    X_train = df_train[symptom_cols].values
    y_train_raw = df_train['prognosis'].values

    X_test  = df_test[symptom_cols].values
    y_test_raw  = df_test['prognosis'].values

    le = LabelEncoder()
    le.fit(np.concatenate([y_train_raw, y_test_raw]))
    y_train = le.transform(y_train_raw)
    y_test  = le.transform(y_test_raw)

    return X_train, y_train, X_test, y_test, symptom_cols, le


def train_and_evaluate(X_train, y_train, X_test, y_test, le):
    """Train all 3 models and return results dict."""
    models = {
        'Naive Bayes':    GaussianNB(),
        'Decision Tree':  DecisionTreeClassifier(random_state=42, max_depth=20),
        'Random Forest':  RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

        results[name] = {
            'model':    model,
            'accuracy': acc,
            'cv_mean':  cv_scores.mean(),
            'cv_std':   cv_scores.std(),
            'y_pred':   y_pred,
            'report':   classification_report(y_test, y_pred,
                            target_names=le.classes_, output_dict=True),
        }
        print(f"  → Test Accuracy: {acc*100:.2f}%  |  CV: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    return results


def save_models(results, symptom_cols, le, best_name):
    """Pickle models, symptom list, label encoder."""
    # Save each model
    for name, res in results.items():
        safe = name.lower().replace(' ', '_')
        path = os.path.join(MODELS_DIR, f'{safe}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(res['model'], f)
        print(f"  Saved {path}")

    # Save symptom columns
    with open(os.path.join(MODELS_DIR, 'symptoms.pkl'), 'wb') as f:
        pickle.dump(symptom_cols, f)

    # Save label encoder
    with open(os.path.join(MODELS_DIR, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)

    # Save best model name
    with open(os.path.join(MODELS_DIR, 'best_model.txt'), 'w') as f:
        f.write(best_name)

    # Save accuracy summary JSON-like
    summary = {n: {'accuracy': r['accuracy'], 'cv_mean': r['cv_mean']}
               for n, r in results.items()}
    with open(os.path.join(MODELS_DIR, 'accuracy_summary.pkl'), 'wb') as f:
        pickle.dump(summary, f)

    print("\n  All models saved.")


def generate_charts(results, y_test, le):
    """Generate and save evaluation charts."""
    # ── 1. Model Accuracy Comparison Bar Chart ────────────────────────────
    names  = list(results.keys())
    accs   = [r['accuracy'] * 100 for r in results.values()]
    cv_means = [r['cv_mean'] * 100 for r in results.values()]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    bars1 = ax.bar(x - 0.2, accs, 0.35, label='Test Accuracy', color='#2563eb', alpha=0.85)
    bars2 = ax.bar(x + 0.2, cv_means, 0.35, label='CV Accuracy (5-fold)', color='#16a34a', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(80, 102)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'model_comparison.png'), dpi=120)
    plt.close()
    print("  Chart: model_comparison.png")

    # ── 2. Confusion Matrix for best model (Random Forest) ────────────────
    best_res = results['Random Forest']
    cm = confusion_matrix(y_test, best_res['y_pred'])

    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_,
                ax=ax, linewidths=0.3, annot_kws={'size': 7})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Random Forest — Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'confusion_matrix.png'), dpi=100)
    plt.close()
    print("  Chart: confusion_matrix.png")

    # ── 3. Top 20 Most Common Diseases Distribution ───────────────────────
    df_train = pd.read_csv(os.path.join(MODELS_DIR, 'Training.csv'))
    disease_counts = df_train['prognosis'].value_counts().head(20)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(disease_counts)))
    bars = ax.barh(disease_counts.index[::-1], disease_counts.values[::-1], color=colors[::-1])
    ax.set_xlabel('Number of Training Samples', fontsize=11)
    ax.set_title('Top 20 Diseases — Training Data Distribution', fontsize=13, fontweight='bold')
    for bar in bars:
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{int(bar.get_width())}', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'disease_distribution.png'), dpi=120)
    plt.close()
    print("  Chart: disease_distribution.png")

    # ── 4. Per-model F1 score chart (macro avg) ───────────────────────────
    f1_scores = [r['report']['macro avg']['f1-score'] * 100 for r in results.values()]
    precision = [r['report']['macro avg']['precision'] * 100 for r in results.values()]
    recall    = [r['report']['macro avg']['recall'] * 100 for r in results.values()]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    ax.plot(x, f1_scores, 'o-', color='#dc2626', label='F1-Score', linewidth=2, markersize=8)
    ax.plot(x, precision, 's-', color='#2563eb', label='Precision', linewidth=2, markersize=8)
    ax.plot(x, recall,    '^-', color='#16a34a', label='Recall', linewidth=2, markersize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(80, 102)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Model Performance Metrics (Macro Avg)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'performance_metrics.png'), dpi=120)
    plt.close()
    print("  Chart: performance_metrics.png")


def main():
    print("=" * 55)
    print("  Healthcare Disease Prediction — Model Training")
    print("=" * 55)

    print("\n[1/4] Loading data...")
    df_train, df_test = load_data()
    print(f"  Train: {df_train.shape} | Test: {df_test.shape}")
    print(f"  Diseases: {df_train['prognosis'].nunique()} | Symptoms: {len(df_train.columns)-1}")

    print("\n[2/4] Preparing features...")
    X_train, y_train, X_test, y_test, symptom_cols, le = prepare_features(df_train, df_test)
    print(f"  Symptoms: {len(symptom_cols)} | Classes: {len(le.classes_)}")

    print("\n[3/4] Training models...")
    results = train_and_evaluate(X_train, y_train, X_test, y_test, le)

    # Pick best
    best_name = max(results, key=lambda n: results[n]['accuracy'])
    print(f"\n  Best model: {best_name} ({results[best_name]['accuracy']*100:.2f}%)")

    print("\n[4/4] Saving models & charts...")
    save_models(results, symptom_cols, le, best_name)
    generate_charts(results, y_test, le)

    print("\n✅ Training complete!")
    print(f"   Charts saved to: {CHARTS_DIR}")
    print(f"   Models saved to: {MODELS_DIR}")


if __name__ == '__main__':
    main()
