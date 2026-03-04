"""
main.py
-------
End-to-end pipeline entry point.

Usage
-----
    # Run with synthetic data (default)
    python main.py

    # Run with the real UCI CSV
    python main.py --data data/bank-additional-full.csv

    # Skip hyperparameter tuning (faster)
    python main.py --no-tune
"""

import argparse
import os
import sys

import pandas as pd

from src.data_generator import generate_bank_data, load_uci_data
from src.preprocessor   import preprocess
from src.model          import train_decision_tree, cross_validate_model
from src.evaluate       import evaluate_model, plot_dashboard


def parse_args():
    parser = argparse.ArgumentParser(
        description='Bank Marketing Decision Tree Classifier')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to UCI bank-additional-full.csv (optional)')
    parser.add_argument('--n', type=int, default=10_000,
                        help='Rows to generate if no --data provided (default 10000)')
    parser.add_argument('--no-tune', action='store_true',
                        help='Skip GridSearchCV and use default params')
    parser.add_argument('--output', type=str,
                        default='results/bank_decision_tree_results.png',
                        help='Path to save the visualisation dashboard')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs('results', exist_ok=True)

    print("=" * 65)
    print("  DECISION TREE CLASSIFIER — UCI BANK MARKETING DATASET")
    print("=" * 65)

    # ── 1. Load / generate data ──────────────────────────────────────
    if args.data:
        print(f"\n[1] Loading UCI data from: {args.data}")
        df = load_uci_data(args.data)
    else:
        print(f"\n[1] Generating synthetic dataset ({args.n:,} rows) …")
        df = generate_bank_data(n=args.n)

    print(f"    Shape         : {df.shape}")
    print(f"    Class balance : {df['y'].value_counts().to_dict()}")
    print(f"    Purchase rate : {(df['y']=='yes').mean()*100:.1f}%")

    # ── 2. Preprocess ────────────────────────────────────────────────
    print("\n[2] Preprocessing …")
    X_train, X_test, y_train, y_test, feature_names = preprocess(df)
    print(f"    Train (balanced) : {y_train.value_counts().to_dict()}")
    print(f"    Test             : {y_test.value_counts().to_dict()}")

    # ── 3. Train model ───────────────────────────────────────────────
    print("\n[3] Training Decision Tree …")
    model, best_params = train_decision_tree(
        X_train, y_train, tune=not args.no_tune)
    print(f"    Nodes  : {model.tree_.node_count}")
    print(f"    Leaves : {model.tree_.n_leaves}")

    # ── 4. Cross-validate ────────────────────────────────────────────
    print("\n[4] Cross-validation (ROC-AUC, 5-fold) …")
    cv_scores = cross_validate_model(model, X_train, y_train)
    print(f"    CV AUC : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── 5. Evaluate on test set ──────────────────────────────────────
    print("\n[5] Evaluating on test set …")
    metrics = evaluate_model(model, X_test, y_test)

    # ── 6. Visualise ─────────────────────────────────────────────────
    print("\n[6] Generating visualisation dashboard …")
    plot_dashboard(
        model=model, df_raw=df,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        feature_names=feature_names,
        cv_scores=cv_scores,
        save_path=args.output
    )

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  Best params  : {best_params}")
    print(f"  Accuracy     : {metrics['accuracy']:.4f}")
    print(f"  Precision    : {metrics['precision']:.4f}")
    print(f"  Recall       : {metrics['recall']:.4f}")
    print(f"  F1-Score     : {metrics['f1']:.4f}")
    print(f"  ROC-AUC      : {metrics['roc_auc']:.4f}")
    print(f"  CV AUC       : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("=" * 65)

    top5 = pd.Series(model.feature_importances_,
                     index=feature_names).nlargest(5)
    print("\n  Top 5 Features:")
    for rank, (feat, imp) in enumerate(top5.items(), 1):
        print(f"    {rank}. {feat:<22}  {imp:.4f}")

    print(f"\n✅ Done! Chart saved to: {args.output}\n")


if __name__ == '__main__':
    main()
