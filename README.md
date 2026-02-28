# SCT-DS-03
Decision Tree Classifier - UCI Bank Marketing
# 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Visualizations](#-visualizations)
- [Contributing](#-contributing)
- [License](#-license)
- [Citation](#-citation)

---

## 🔍 Overview

This project implements a full, production-style machine learning pipeline to classify whether a bank client will subscribe to a **term deposit** product based on demographic and behavioral data.

### Pipeline Steps

| Step | Description |
|------|-------------|
| 1. Data | Generates synthetic data matching UCI Bank Marketing schema (or load real CSV) |
| 2. Preprocessing | Label encoding, stratified train/test split, minority-class oversampling |
| 3. Tuning | `GridSearchCV` over depth, split size, leaf size, and criterion |
| 4. Training | Final `DecisionTreeClassifier` with `class_weight='balanced'` |
| 5. Evaluation | Accuracy, Precision, Recall, F1, ROC-AUC, 5-fold CV |
| 6. Visualization | 9-panel dashboard saved as PNG |

---

## 📊 Dataset

**Source:** [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)

The dataset contains results of direct marketing campaigns (phone calls) of a Portuguese banking institution. The goal is to predict if the client will subscribe to a term deposit (`y = yes/no`).

| Property | Value |
|----------|-------|
| Full dataset size | 45,211 records |
| Features | 20 input + 1 target |
| Target | Binary: `yes` / `no` |
| Positive class rate | ~11.3% (class imbalanced) |
| Format | Semicolon-separated CSV |

### Feature Groups

<details>
<summary><b>📁 Client Demographics (7 features)</b></summary>

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numeric | Client age in years |
| `job` | Categorical | Type of job (12 categories) |
| `marital` | Categorical | Marital status: married / single / divorced |
| `education` | Categorical | Education level (7 categories) |
| `default` | Categorical | Has credit in default? yes / no / unknown |
| `housing` | Categorical | Has housing loan? yes / no / unknown |
| `loan` | Categorical | Has personal loan? yes / no / unknown |

</details>

<details>
<summary><b>📞 Campaign Information (8 features)</b></summary>

| Feature | Type | Description |
|---------|------|-------------|
| `contact` | Categorical | Contact type: cellular / telephone |
| `month` | Categorical | Last contact month of year |
| `day_of_week` | Categorical | Last contact day of week |
| `duration` | Numeric | Last contact duration in seconds |
| `campaign` | Numeric | Number of contacts in this campaign |
| `pdays` | Numeric | Days since last contact (999 = not previously contacted) |
| `previous` | Numeric | Number of contacts before this campaign |
| `poutcome` | Categorical | Outcome of previous campaign |

</details>

<details>
<summary><b>📈 Macroeconomic Indicators (5 features)</b></summary>

| Feature | Type | Description |
|---------|------|-------------|
| `emp_var_rate` | Numeric | Employment variation rate (quarterly) |
| `cons_price_idx` | Numeric | Consumer price index (monthly) |
| `cons_conf_idx` | Numeric | Consumer confidence index (monthly) |
| `euribor3m` | Numeric | Euribor 3-month rate (daily) |
| `nr_employed` | Numeric | Number of employees (quarterly) |

</details>

---

## 📁 Project Structure

```
bank-marketing-decision-tree/
│
├── 📄 README.md                         ← You are here
├── 📄 requirements.txt                  ← Python dependencies
├── 📄 .gitignore
├── 📄 LICENSE                           ← MIT
│
├── 🐍 main.py                           ← End-to-end pipeline entry point
│
├── 📂 src/
│   ├── __init__.py
│   ├── data_generator.py                ← Synthetic data + UCI CSV loader
│   ├── preprocessor.py                  ← Encoding, splitting, oversampling
│   ├── model.py                         ← GridSearchCV tuning + training
│   └── evaluate.py                      ← Metrics + 9-panel dashboard
│
├── 📂 notebooks/
│   └── bank_marketing_analysis.ipynb    ← Full interactive walkthrough
│
├── 📂 data/
│   └── README.md                        ← How to download the UCI dataset
│
├── 📂 results/
│   └── bank_decision_tree_results.png   ← Output visualization dashboard
│
└── 📂 tests/
    ├── test_preprocessor.py             ← Unit tests for preprocessing
    └── test_model.py                    ← Unit tests for model & evaluation
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.8 or higher
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/bank-marketing-decision-tree.git
cd bank-marketing-decision-tree
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Create environment
python -m venv venv

# Activate — macOS/Linux
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the Full Pipeline (Synthetic Data)

```bash
python main.py
```

### Run with Real UCI Dataset

1. Download `bank-additional-full.csv` from the [UCI Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing)
2. Place it in the `data/` folder
3. Run:

```bash
python main.py --data data/bank-additional-full.csv
```

### Options

```bash
python main.py --help

# Options:
#   --data PATH     Path to bank-additional-full.csv (uses synthetic data if omitted)
#   --n INT         Number of rows to generate synthetically (default: 10000)
#   --no-tune       Skip GridSearchCV and use default hyperparameters (faster)
#   --output PATH   Where to save the visualization dashboard
```

### Use as a Python Module

```python
from src.data_generator import generate_bank_data
from src.preprocessor   import preprocess
from src.model          import train_decision_tree, cross_validate_model
from src.evaluate       import evaluate_model, plot_dashboard

# 1. Load data
df = generate_bank_data(n=10_000)

# 2. Preprocess
X_train, X_test, y_train, y_test, feature_names = preprocess(df)

# 3. Train
model, best_params = train_decision_tree(X_train, y_train, tune=True)

# 4. Evaluate
metrics = evaluate_model(model, X_test, y_test)
print(metrics)
# {'accuracy': 0.78, 'precision': 0.18, 'recall': 0.44, 'f1': 0.25, 'roc_auc': 0.60}

# 5. Cross-validate
cv_scores = cross_validate_model(model, X_train, y_train)
print(f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

### Run Unit Tests

```bash
python -m pytest tests/ -v
```

---

## 📈 Results

### Best Hyperparameters (5-fold GridSearchCV)

| Parameter | Value |
|-----------|-------|
| Criterion | Gini impurity |
| Max Depth | 10 |
| Min Samples Split | 10 |
| Min Samples Leaf | 5 |
| Class Weight | Balanced |
| Tree Nodes | 431 |
| Tree Leaves | 216 |

### Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 0.778 |
| **Precision** (Purchase) | 0.177 |
| **Recall** (Purchase) | 0.439 |
| **F1-Score** (Purchase) | 0.253 |
| **ROC-AUC** | 0.598 |
| **CV AUC (5-fold)** | **0.880 ± 0.003** |

> ⚠️ **Note on class imbalance:** The dataset has ~8–11% positive rate. Accuracy alone is misleading — Recall and ROC-AUC are more meaningful metrics for this business problem.

### Top 5 Most Important Features

| Rank | Feature | Importance | Description |
|------|---------|-----------|-------------|
| 🥇 1 | `duration` | 0.284 | Call duration — strongest predictor |
| 🥈 2 | `emp_var_rate` | 0.140 | Employment variation rate |
| 🥉 3 | `cons_price_idx` | 0.100 | Consumer price index |
| 4 | `poutcome` | 0.097 | Previous campaign outcome |
| 5 | `cons_conf_idx` | 0.083 | Consumer confidence index |

> 💡 **Key Insight:** Call duration is the single strongest predictor — longer calls strongly correlate with successful subscriptions. However, duration is only known *after* the call, making it a post-hoc feature. In a real deployment scenario, models excluding `duration` are more actionable for pre-call targeting.

---

## 🖼️ Visualizations

The pipeline generates a **9-panel dashboard** saved to `results/bank_decision_tree_results.png`:

| Panel | Title | Description |
|-------|-------|-------------|
| **A** | Class Distribution | Bar chart of yes vs. no counts |
| **B** | Age Distribution | Histogram split by outcome |
| **C** | Purchase Rate by Job | Horizontal bar chart per job type |
| **D** | Confusion Matrix | TP / TN / FP / FN heatmap |
| **E** | ROC Curve | With AUC score and random baseline |
| **F** | Feature Importances | Top 15 features ranked |
| **G** | Decision Tree | Full tree visualization at depth=3 |
| **H** | CV AUC Scores | 5-fold cross-validation bar chart |
| **I** | Age vs. Duration | Scatter plot colored by outcome |
| **J** | Performance Summary | Metrics summary card |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

```bash
# 1. Fork the repo
# 2. Create your feature branch
git checkout -b feature/my-new-feature

# 3. Commit your changes
git commit -m "Add: my new feature"

# 4. Push to the branch
git push origin feature/my-new-feature

# 5. Open a Pull Request
```

Please make sure to:
- Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines
- Add unit tests for any new functionality in `tests/`
- Update this README if needed

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 📚 Citation

If you use this project or the UCI dataset, please cite:

```bibtex
@article{moro2014data,
  title     = {A data-driven approach to predict the success of bank telemarketing},
  author    = {Moro, S{\'e}rgio and Cortez, Paulo and Rita, Paulo},
  journal   = {Decision Support Systems},
  volume    = {62},
  pages     = {22--31},
  year      = {2014},
  publisher = {Elsevier},
  doi       = {10.1016/j.dss.2014.03.001}
}
```

---

## 🙏 Acknowledgements

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) for the dataset
- [scikit-learn](https://scikit-learn.org/) for the ML framework
- Moro et al. (2014) for the original research and dataset

---

<div align="center">
  Made with ❤️ | ⭐ Star this repo if you found it helpful!
</div>
