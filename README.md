# Spam Detection Using Decision Trees

A machine learning project that trains and evaluates Decision Tree classifiers on the UCI Spambase dataset. The project examines the effects of feature normalization, overfitting controls, cost-complexity pruning, and class-imbalance strategies on email spam classification.

## Dataset

The repository includes the UCI Spambase dataset and its accompanying metadata files.

- 4,601 email samples
- 57 numerical input features
- One binary target column: spam or non-spam
- Dataset shape: 4,601 rows × 58 columns

The data is divided using stratified splitting to preserve class proportions:

- 60% training
- 20% validation
- 20% testing

## Experiments

### Normalization Study

- No normalization
- Min-Max normalization
- Standard normalization

### Overfitting and Pruning

- Unconstrained Decision Tree
- Maximum-depth comparison
- Cost-complexity pruning using `ccp_alpha`

### Class-Imbalance Study

- Baseline imbalanced training data
- Class weighting using `class_weight='balanced'`
- Random oversampling of the minority class

### Additional Analysis

- Gini impurity versus entropy comparison
- Confusion matrices and classification reports
- Decision Tree visualization and exported rules
- Feature-importance analysis
- Performance comparison across experimental scenarios

All experiments use a fixed random state for reproducibility. Feature scalers are fitted exclusively on the training data and then applied to the validation and test sets to prevent data leakage.

## Results

The no-normalization model achieved **91.31% test accuracy**, with **0.87 recall** and an **0.89 F1-score** for the spam class.

Random oversampling increased spam recall to **0.88** and achieved an **0.88 F1-score**, providing the strongest spam-detection performance among the evaluated class-balancing strategies.

## Repository Contents

- `spam_detection_project.ipynb` — complete data preparation, model training, experiments, evaluation, and visualizations.
- `spambase/` — dataset and accompanying documentation.
- `decision_tree_rules.txt` — exported rules from the selected Decision Tree model.
- PNG files — saved class-distribution, confusion-matrix, pruning, feature-importance, and performance-summary visualizations.

## Requirements

- Python 3
- Jupyter Notebook or JupyterLab
- NumPy
- pandas
- Matplotlib
- seaborn
- scikit-learn

Install the required packages:

```bash
pip install jupyter numpy pandas matplotlib seaborn scikit-learn
```

## How to Run

Clone the repository and enter its directory:

```bash
git clone https://github.com/r26th/spam-detection-decision-tree.git
cd spam-detection-decision-tree
```

Launch the notebook:

```bash
jupyter notebook spam_detection_project.ipynb
```

Run the notebook cells in order to reproduce the data preparation, trained models, evaluation metrics, and generated visualizations.
