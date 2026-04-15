# Getting Started

## Installation

GlassBox requires **Python 3.11+** and **NumPy**.

=== "Runtime only"

    ```bash
    pip install .
    ```

=== "With dev tools"

    ```bash
    pip install .[dev]
    ```

---

## Quickstart

A minimal end-to-end workflow: **load → inspect → clean → train → predict**.

### 1. Load a Dataset

```python
from glassbox.frame import read_csv

ds = read_csv("data.csv")
print(ds)  # Dataset(shape=(1000, 12), columns=[...])
```

### 2. Inspect the Data

```python
from glassbox.inspector import DataAuditor

auditor = DataAuditor()
report = auditor.run_audit(ds)

# Feature types detected automatically
print(report.feature_types)

# Outlier counts per numeric column
print(report.outliers_info)

# Export to JSON
print(report.to_json())
```

### 3. Clean the Data

```python
import numpy as np
from glassbox.cleaner import SimpleImputer, StandardScaler

X = ds.get_columns(["feat_1", "feat_2"]).data.astype(float)

# Impute missing values with the column mean
imputer = SimpleImputer()
X = imputer.fit_transform(X)

# Standardize to zero-mean, unit-variance
scaler = StandardScaler()
X = scaler.fit_transform(X)
```

### 4. Train a Model

```python
from glassbox.models import DecisionTreeClassifier

y = ds.get_columns("target").data[:, 0].astype(float)

model = DecisionTreeClassifier(max_depth=10)
model.fit(X, y)
```

### 5. Predict

```python
predictions = model.predict(X)
print(predictions[:10])
```

---

## What's Next?

- **[Frame](user-guide/frame.md)** — Data loading and manipulation.
- **[Inspector](user-guide/inspector.md)** — Exploratory Data Analysis.
- **[Cleaner](user-guide/cleaner.md)** — Data preprocessing pipeline.
- **[Models](user-guide/models.md)** — Machine learning algorithms.
- **[API Reference](reference/index.md)** — Auto-generated from source docstrings.
