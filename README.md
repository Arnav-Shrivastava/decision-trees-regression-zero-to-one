# Decision Trees: From Zero to Confident

A comprehensive, beginner-friendly learning project and hands-on masterclass on **Decision Trees** for classification and regression. This project covers theoretical intuition, mathematical derivations, scratch implementations, visual explanations, pruning techniques, model instability, feature importance subtleties, and regression trees.

---

## 📌 Project Overview
- **Domain**: Automated Loan Approval Prediction
- **Input Features**: Annual Income, Credit Score, Debt-to-Income Ratio, Employment Years, Loan Amount requested
- **Target**: Loan Approval Status (`1 = Approved`, `0 = Denied`)
- **Key Concepts Covered**:
  1. Splitting criteria (Gini Impurity, Entropy, Information Gain)
  2. Building a classification tree from scratch using pure NumPy & Pandas
  3. Visualizing decision boundaries and explaining individual predictions to non-technical stakeholders
  4. Overfitting analysis (variance vs. bias, depth curves)
  5. Tree pruning (pre-pruning constraints vs. post-pruning Cost-Complexity / Minimal $\alpha$)
  6. High variance and instability under bootstrap resampling (the core motivation for Random Forests)
  7. Feature importance nuances: MDI (Gini importance) vs. Permutation Importance and cardinality bias
  8. Decision tree regression and piecewise step-function behavior
  9. Exercises, verification tests, and practical challenge problems

---

## 📂 Project Structure
```text
decision-trees-regression-zero-to-one/
│
├── decision_trees_zero_to_one.ipynb  # Primary interactive walkthrough & master notebook
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git exclusion rules
└── README.md                         # Project documentation and summary
```

---

## 🚀 Getting Started

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/Arnav-Shrivastava/decision-trees-regression-zero-to-one.git
cd decision-trees-regression-zero-to-one
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch the Jupyter Notebook
```bash
jupyter notebook decision_trees_zero_to_one.ipynb
```

---

*Detailed results, training curves, tree visualizations, and exercise solutions will be populated as each module runs.*
