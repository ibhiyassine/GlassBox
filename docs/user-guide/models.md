# Models

The `glassbox.models` module provides machine learning algorithms for **classification** and **regression**. All models follow the **fit → predict** contract defined by `BaseModel`.

---

## Model API

```python
model.fit(X, y)       # Train on (n_samples, n_features) array
model.predict(X)      # Returns predictions array
```

---

## Decision Trees

CART-style decision trees that recursively split features to minimize a cost function.

### DecisionTreeClassifier

Uses **Gini impurity** as the split criterion and **majority vote** (mode) for leaf predictions.

```python
from glassbox.models import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=10, min_samples_split=5)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

### DecisionTreeRegressor

Uses **variance reduction** as the split criterion and **mean** for leaf predictions.

```python
from glassbox.models import DecisionTreeRegressor

model = DecisionTreeRegressor(max_depth=15)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `max_depth` | `100` | Maximum depth of the tree. |
| `min_samples_split` | `2` | Minimum samples needed to split a node. |

---

## Random Forests

Ensemble of decision trees trained on bootstrapped samples with random feature subsets (√n_features).

### RandomForestClassifier

Aggregates predictions via **majority vote**.

```python
from glassbox.models import RandomForestClassifier

model = RandomForestClassifier(n_estimators=50, max_depth=10)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

### RandomForestRegressor

Aggregates predictions via **averaging**.

```python
from glassbox.models import RandomForestRegressor

model = RandomForestRegressor(n_estimators=50, max_depth=10)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `n_estimators` | `100` | Number of trees in the forest. |
| `max_depth` | `100` | Maximum depth of each tree. |
| `min_samples_split` | `2` | Minimum samples needed to split a node. |

---

## K-Nearest Neighbors

Instance-based learning that predicts based on the `k` closest training samples.

### Configuration Enums

=== "Distance Metrics"

    | `DistanceMetric` | Formula |
    |---|---|
    | `EUCLIDEAN` | √Σ(xᵢ − yᵢ)² |
    | `MANHATTAN` | Σ\|xᵢ − yᵢ\| |

=== "Search Algorithms"

    | `SearchAlgorithm` | Description |
    |---|---|
    | `BRUTE_FORCE` | Exhaustive pairwise distance computation. |
    | `KD_TREE` | Space-partitioning tree for faster lookup. |

### KNeighborsClassifier

Predicts via **majority vote** among the k nearest neighbors.

```python
from glassbox.models import KNeighborsClassifier, DistanceMetric, SearchAlgorithm

model = KNeighborsClassifier(
    k=5,
    metric=DistanceMetric.EUCLIDEAN,
    algorithm=SearchAlgorithm.KD_TREE,
)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

### KNeighborsRegressor

Predicts via **averaging** the k nearest neighbors' targets.

```python
from glassbox.models import KNeighborsRegressor

model = KNeighborsRegressor(k=7, metric=DistanceMetric.MANHATTAN)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `k` | `5` | Number of neighbors. |
| `metric` | `EUCLIDEAN` | Distance metric. |
| `algorithm` | `BRUTE_FORCE` | Nearest-neighbor search strategy. |

??? tip "Single-sample prediction"
    KNN models accept both batch input `(n_samples, n_features)` and single-sample input `(n_features,)` in `predict()`.

---

## Gaussian Naive Bayes

A probabilistic classifier based on applying Bayes' theorem with strong (naive) independence assumptions between the features. Features are assumed to follow a Gaussian distribution.

### GaussianNB

```python
from glassbox.models import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

---

## Linear Models

Models that fit a linear surface to the data, trained using gradient descent optimization.

### LinearRegression

Predicts a continuous target variable by finding the line of best fit.

```python
from glassbox.models import LinearRegression

model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

### LogisticRegression

Predicts a categorical target variable using the logistic (sigmoid) function to output probabilities.

```python
from glassbox.models import LogisticRegression

model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `learning_rate` | `0.01` | The step size for gradient descent optimization. |
| `n_iterations` | `1000` | The number of optimization iterations. |

---

## API Reference

::: glassbox.models
    options:
      show_root_heading: false
      members:
        - BaseModel
        - DecisionTreeClassifier
        - DecisionTreeRegressor
        - RandomForestClassifier
        - RandomForestRegressor
        - KNeighborsClassifier
        - KNeighborsRegressor
        - DistanceMetric
        - SearchAlgorithm
        - GaussianNB
        - BaseLinearModel
        - LinearRegression
        - LogisticRegression
        - LearningSchedule
