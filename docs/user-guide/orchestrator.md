# Orchestrator

The `glassbox.orchestrator` module provides tools for **model selection** and **hyperparameter tuning** through cross-validation.

---

## Hyperparameter Search

Automate the process of finding the optimal hyperparameters for your models.

### GridSearchCV

Exhaustive search over specified parameter values for an estimator. Each combination is evaluated using cross-validation.

```python
from glassbox.orchestrator import GridSearchCV
from glassbox.models import DecisionTreeClassifier

param_grid = {
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5]
}

model = DecisionTreeClassifier()
search = GridSearchCV(model, param_grid, cv=5)
search.fit(X_train, y_train)

# The best model is automatically refitted on the entire training set
print(search.best_params_)
preds = search.predict(X_test)
```

### RandomizedSearchCV

Randomized search over a hyperparameter space. Trades exhaustiveness for computational speed.

```python
from glassbox.orchestrator import RandomizedSearchCV
from glassbox.models import RandomForestClassifier

param_space = {
    'max_depth': [5, 10, 15, 20],
    'n_estimators': [10, 50, 100]
}

model = RandomForestClassifier()
search = RandomizedSearchCV(model, param_space, n_iter=10, cv=5)
search.fit(X_train, y_train)

print(search.best_params_)
preds = search.predict(X_test)
```

---

## Cross-Validation Splitters

Strategies to split data into training and validation sets.

### KFoldSplitter

Splits the dataset into `n_splits` consecutive folds, preserving the underlying order if not shuffled.

```python
from glassbox.orchestrator import KFoldSplitter

splitter = KFoldSplitter(n_splits=5, shuffle=True, random_seed=42)
for train_idx, val_idx in splitter.split(X):
    X_fold_train, X_fold_val = X[train_idx], X[val_idx]
```

### StratifiedKFoldSplitter

Splits the dataset into folds while preserving the percentage of samples for each class in `y`. Ideally suited for classification problems with imbalanced labels.

```python
from glassbox.orchestrator import StratifiedKFoldSplitter

splitter = StratifiedKFoldSplitter(n_splits=5, shuffle=True, random_seed=42)
for train_idx, val_idx in splitter.split(X, y):
    y_fold_train, y_fold_val = y[train_idx], y[val_idx]
```

---

## API Reference

::: glassbox.orchestrator
    options:
      show_root_heading: false
      members:
        - BaseSearch
        - BaseSplitter
        - GridSearchCV
        - RandomizedSearchCV
        - KFoldSplitter
        - StratifiedKFoldSplitter
