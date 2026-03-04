"""
evaluate.py
-----------
Model evaluation, metrics computation, and visualisation dashboard.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay, accuracy_score,
    precision_score, recall_score, f1_score
)
from typing import Dict, List, Any


# ── Colour palette ──────────────────────────────────────────────────────────
BLUE   = '#2980B9'
GREEN  = '#27AE60'
RED    = '#E74C3C'
ORANGE = '#E67E22'
PURPLE = '#8E44AD'


def compute_metrics(
    model: DecisionTreeClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Compute a full set of classification metrics.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall':    recall_score(y_test, y_pred, zero_division=0),
        'f1':        f1_score(y_test, y_pred, zero_division=0),
        'roc_auc':   roc_auc_score(y_test, y_proba),
    }


def evaluate_model(
    model: DecisionTreeClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    verbose: bool = True
) -> Dict[str, float]:
    """Print classification report and return metrics dict."""
    y_pred = model.predict(X_test)
    metrics = compute_metrics(model, X_test, y_test)

    if verbose:
        print("\n  Classification Report:")
        print(classification_report(y_test, y_pred,
              target_names=['No Purchase', 'Purchase']))
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")

    return metrics


def plot_dashboard(
    model: DecisionTreeClassifier,
    df_raw: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: List[str],
    cv_scores: np.ndarray,
    save_path: str = "results/bank_decision_tree_results.png"
) -> None:
    """
    Generate a 9-panel visualisation dashboard and save to disk.

    Panels
    ------
    A – Class distribution
    B – Age distribution by outcome
    C – Purchase rate by job
    D – Confusion matrix
    E – ROC curve
    F – Feature importances (top 15)
    G – Decision tree (depth 3)
    H – 5-fold CV AUC scores
    I – Age vs call duration scatter
    J – Performance summary card
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(model, X_test, y_test)
    auc     = metrics['roc_auc']

    fig = plt.figure(figsize=(22, 28))
    fig.patch.set_facecolor('#F8F9FA')

    ax_title = fig.add_axes([0, 0.97, 1, 0.03])
    ax_title.axis('off')
    ax_title.text(0.5, 0.5,
        'Decision Tree Classifier  ·  UCI Bank Marketing Dataset',
        ha='center', va='center', fontsize=18, fontweight='bold', color='#2C3E50')

    # ── A: Class distribution ──────────────────
    ax1 = fig.add_subplot(4, 3, 1)
    counts = df_raw['y'].value_counts()
    bars   = ax1.bar(['No Purchase', 'Purchase'], counts.values,
                      color=[RED, GREEN], edgecolor='white', linewidth=1.5, width=0.5)
    for b in bars:
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()+80,
                 f'{b.get_height():,}', ha='center', fontsize=11, fontweight='bold')
    ax1.set_title('A  Class Distribution', fontsize=12, fontweight='bold', pad=8)
    ax1.set_ylabel('Count'); ax1.set_ylim(0, counts.max() * 1.15)
    ax1.spines[['top','right']].set_visible(False); ax1.set_facecolor('#FDFEFE')

    # ── B: Age distribution ────────────────────
    ax2 = fig.add_subplot(4, 3, 2)
    for label, color in [('no', RED), ('yes', GREEN)]:
        ax2.hist(df_raw[df_raw['y']==label]['age'], bins=25, alpha=0.65,
                 color=color, edgecolor='white', label=label.capitalize())
    ax2.set_title('B  Age Distribution by Outcome', fontsize=12, fontweight='bold', pad=8)
    ax2.set_xlabel('Age'); ax2.set_ylabel('Frequency')
    ax2.legend(title='Purchase'); ax2.spines[['top','right']].set_visible(False)
    ax2.set_facecolor('#FDFEFE')

    # ── C: Purchase rate by job ────────────────
    ax3 = fig.add_subplot(4, 3, 3)
    job_rate = df_raw.groupby('job')['y'].apply(
        lambda x: (x=='yes').mean() * 100).sort_values()
    avg_rate = (df_raw['y']=='yes').mean() * 100
    colors_c = [GREEN if v > avg_rate else BLUE for v in job_rate.values]
    job_rate.plot(kind='barh', ax=ax3, color=colors_c, edgecolor='white')
    ax3.axvline(x=avg_rate, color=RED, linestyle='--', lw=1.5, label='Overall avg')
    ax3.set_title('C  Purchase Rate by Job', fontsize=12, fontweight='bold', pad=8)
    ax3.set_xlabel('Purchase Rate (%)'); ax3.legend(fontsize=9)
    ax3.spines[['top','right']].set_visible(False); ax3.set_facecolor('#FDFEFE')

    # ── D: Confusion matrix ────────────────────
    ax4 = fig.add_subplot(4, 3, 4)
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['No Purchase','Purchase'])
    disp.plot(ax=ax4, colorbar=False, cmap='Blues')
    ax4.set_title('D  Confusion Matrix', fontsize=12, fontweight='bold', pad=8)
    ax4.set_facecolor('#FDFEFE')

    # ── E: ROC curve ───────────────────────────
    ax5 = fig.add_subplot(4, 3, 5)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax5.plot(fpr, tpr, color=BLUE, lw=2, label=f'AUC = {auc:.3f}')
    ax5.plot([0,1],[0,1], 'k--', lw=1, alpha=0.5, label='Random')
    ax5.fill_between(fpr, tpr, alpha=0.12, color=BLUE)
    ax5.set_title('E  ROC Curve', fontsize=12, fontweight='bold', pad=8)
    ax5.set_xlabel('False Positive Rate'); ax5.set_ylabel('True Positive Rate')
    ax5.legend(); ax5.spines[['top','right']].set_visible(False); ax5.set_facecolor('#FDFEFE')

    # ── F: Feature importances ─────────────────
    ax6 = fig.add_subplot(4, 3, 6)
    importances = pd.Series(model.feature_importances_,
                            index=feature_names).sort_values()
    top15   = importances.tail(15)
    color_f = [GREEN if v > importances.quantile(0.8) else BLUE
               for v in top15.values]
    top15.plot(kind='barh', ax=ax6, color=color_f, edgecolor='white')
    ax6.set_title('F  Feature Importances (Top 15)', fontsize=12, fontweight='bold', pad=8)
    ax6.set_xlabel('Importance Score')
    ax6.spines[['top','right']].set_visible(False); ax6.set_facecolor('#FDFEFE')

    # ── G: Tree visualisation (depth 3) ────────
    ax7 = fig.add_subplot(4, 1, 3)
    dt_vis = DecisionTreeClassifier(max_depth=3, class_weight='balanced',
                                    random_state=42)
    dt_vis.fit(X_train, y_train)
    plot_tree(dt_vis, feature_names=feature_names,
              class_names=['No','Yes'], filled=True, rounded=True,
              fontsize=7, ax=ax7, impurity=True, max_depth=3)
    ax7.set_title('G  Decision Tree Visualization (depth = 3)',
                  fontsize=12, fontweight='bold', pad=10)
    ax7.set_facecolor('#FDFEFE')

    # ── H: CV AUC ──────────────────────────────
    ax8 = fig.add_subplot(4, 3, 10)
    ax8.bar(range(1, len(cv_scores)+1), cv_scores,
            color=PURPLE, edgecolor='white', width=0.5)
    ax8.axhline(cv_scores.mean(), color=ORANGE, linestyle='--', lw=2,
                label=f'Mean={cv_scores.mean():.3f}')
    ax8.set_title('H  5-Fold CV AUC Scores', fontsize=12, fontweight='bold', pad=8)
    ax8.set_xlabel('Fold'); ax8.set_ylabel('AUC')
    ax8.set_ylim(0.5, 1.0); ax8.legend()
    ax8.set_xticks(range(1, len(cv_scores)+1))
    ax8.spines[['top','right']].set_visible(False); ax8.set_facecolor('#FDFEFE')

    # ── I: Age vs duration scatter ─────────────
    ax9 = fig.add_subplot(4, 3, 11)
    samp = df_raw.sample(min(1500, len(df_raw)), random_state=1)
    for label, color in [('no', RED), ('yes', GREEN)]:
        sub = samp[samp['y']==label]
        ax9.scatter(sub['age'], sub['duration'], alpha=0.35, s=12,
                    color=color, label=label.capitalize())
    ax9.set_title('I  Age vs Call Duration', fontsize=12, fontweight='bold', pad=8)
    ax9.set_xlabel('Age'); ax9.set_ylabel('Call Duration (s)')
    ax9.legend(title='Purchase')
    ax9.spines[['top','right']].set_visible(False); ax9.set_facecolor('#FDFEFE')

    # ── J: Summary card ────────────────────────
    ax10 = fig.add_subplot(4, 3, 12)
    ax10.axis('off')
    metric_items = [
        ('Accuracy',  f"{metrics['accuracy']:.4f}",  BLUE),
        ('Precision', f"{metrics['precision']:.4f}", GREEN),
        ('Recall',    f"{metrics['recall']:.4f}",    ORANGE),
        ('F1-Score',  f"{metrics['f1']:.4f}",        PURPLE),
        ('ROC-AUC',   f"{auc:.4f}",                  RED),
        ('CV AUC',    f"{cv_scores.mean():.4f} ± {cv_scores.std():.3f}", '#2C3E50'),
    ]
    ax10.text(0.5, 0.97, 'J  Model Performance Summary',
              ha='center', va='top', fontsize=12, fontweight='bold',
              transform=ax10.transAxes)
    for i, (name, val, color) in enumerate(metric_items):
        y_pos = 0.80 - i * 0.13
        rect  = mpatches.FancyBboxPatch(
            (0.05, y_pos-0.04), 0.9, 0.11,
            boxstyle="round,pad=0.01", facecolor=color, alpha=0.15,
            transform=ax10.transAxes, clip_on=False)
        ax10.add_patch(rect)
        ax10.text(0.12, y_pos+0.02, name, transform=ax10.transAxes,
                  fontsize=11, color=color, fontweight='bold', va='center')
        ax10.text(0.88, y_pos+0.02, val, transform=ax10.transAxes,
                  fontsize=11, color='#2C3E50', fontweight='bold',
                  va='center', ha='right')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
    plt.close()
    print(f"  ✓ Dashboard saved → {save_path}")
