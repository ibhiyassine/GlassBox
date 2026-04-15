# Cleaner

The `glassbox.cleaner` module provides scikit-learn-style transformers for data preprocessing. Every transformer follows the **fit → transform** contract defined by `BaseTransformer`.

---

## Transformer API

All transformers share the same interface:

```python
transformer.fit(X)            # Learn parameters from training data
transformer.transform(X)      # Apply the transformation
transformer.fit_transform(X)  # Shorthand: fit + transform
```

Where `X` is always a `np.ndarray` of shape `(n_samples, n_features)`.

---

## Imputation

`SimpleImputer` replaces missing values (`NaN`) using a chosen strategy.

### Available Strategies

| Strategy | Behavior |
|---|---|
| `ImputationStrategy.MEAN` | Replace with column mean (default). |
| `ImputationStrategy.MEDIAN` | Replace with column median. |
| `ImputationStrategy.MODE` | Replace with column mode. |
| `ImputationStrategy.CONSTANT` | Replace with a user-defined constant. |

### Example

```python
from glassbox.cleaner import SimpleImputer, ImputationStrategy

# Mean imputation (default)
imputer = SimpleImputer()
X_clean = imputer.fit_transform(X)

# Constant imputation
imputer = SimpleImputer(
    strategy=ImputationStrategy.CONSTANT,
    constant_value=-1.0,
)
X_clean = imputer.fit_transform(X)
```

---

## Outlier Capping

`OutlierCapper` clips values outside the **IQR bounds** (Q1 − 1.5×IQR, Q3 + 1.5×IQR) to the boundary values.

```python
from glassbox.cleaner import OutlierCapper

capper = OutlierCapper()
X_capped = capper.fit_transform(X)
```

??? info
    NaN values are preserved — only non-missing values are clipped.

---

## Scaling

### StandardScaler

Standardizes features to **zero mean and unit variance**: `z = (x - μ) / σ`.

```python
from glassbox.cleaner import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### MinMaxScaler

Scales features to the **[0, 1] range**: `x' = (x - min) / (max - min)`.

```python
from glassbox.cleaner import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

---

## Encoding

### OneHotEncoder

Converts each categorical column into binary indicator columns — one per unique category.

```python
from glassbox.cleaner import OneHotEncoder

encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X_categorical)
# shape: (n_samples, total_unique_categories)
```

### LabelEncoder

Maps each unique label to an integer `0, 1, 2, …`.

```python
from glassbox.cleaner import LabelEncoder

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y.reshape(-1, 1))
```

??? tip
    Use `LabelEncoder` for target labels and `OneHotEncoder` for input features.

---

## Full Pipeline Example

```python
import numpy as np
from glassbox.cleaner import (
    SimpleImputer,
    OutlierCapper,
    StandardScaler,
    OneHotEncoder,
    LabelEncoder,
)

# Assume X_num is numeric features, X_cat is categorical features

# Numeric pipeline
X_num = SimpleImputer().fit_transform(X_num)
X_num = OutlierCapper().fit_transform(X_num)
X_num = StandardScaler().fit_transform(X_num)

# Categorical pipeline
X_cat = OneHotEncoder().fit_transform(X_cat)

# Combine
X_final = np.hstack([X_num, X_cat])
```

---

## API Reference

::: glassbox.cleaner
    options:
      show_root_heading: false
      members:
        - SimpleImputer
        - ImputationStrategy
        - OutlierCapper
        - StandardScaler
        - MinMaxScaler
        - OneHotEncoder
        - LabelEncoder
