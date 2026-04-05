# GlassBox Development Guidelines

## 1. Philosophy

* Clarity over cleverness
* Explicit over implicit
* Simplicity over premature abstraction
* NumPy is the core computation engine

---

## 2. Code Style

### Naming Conventions

* **Classes**: PascalCase (`StandardScaler`, `LinearRegression`)
* **Functions & variables**: snake_case (`train_test_split`, `compute_mean`)
* **Files**: snake_case (`standard_scaler.py`)
* **Folders**: lowercase plural (`models/`, `preprocessing/`)

---

### General Rules

* One responsibility per function/class
* Prefer small functions (≈ 30 lines max)
* Avoid magic numbers → use named constants
* No hidden transformations inside functions
* No global state (use class attributes instead)

---

### Type Hints (Required)

```python
def fit(X: np.ndarray, y: np.ndarray) -> None:
    ...
```

---

## 3. Data Handling

* Internal format:

  * `X: np.ndarray (n_samples, n_features)`
  * `y: np.ndarray (n_samples,)`
* No pandas inside the core pipeline
* Convert all inputs to NumPy early

---

## 4. API Contracts

### Transformers

```python
fit(X, y=None)
transform(X)
fit_transform(X, y=None)
```

### Models

```python
fit(X, y)
predict(X)
```

* Do not break these interfaces
* Keep behavior explicit and predictable

---

## 5. Docstrings (NumPy Style)

All public functions/classes must include docstrings.

```python
def compute_mean(x: np.ndarray) -> float:
    """
    Compute the mean of a 1D array.

    Parameters
    ----------
    x : np.ndarray
        Input array of shape (n,).

    Returns
    -------
    float
        Mean value.
    """
```

### Rules

* Describe **what**, not how
* Always specify shapes and types
* Use `Notes` for assumptions or math

---

## 6. TODO / FIXME / NOTE

Use consistent tags:

```python
# TODO(username): implement stratified split
# FIXME(username): fails with NaN values
# NOTE(username): assumes numeric input
```

### Rules

* Keep them short and actionable
* Avoid vague TODOs
* Remove them once resolved

---

## 7. Project Structure

* One main class per file
* Avoid generic files like `utils.py`
* Group by responsibility:

```
glassbox/
    inspector/
    preprocessing/
    models/
    evaluation/
    selection/
    core/
```

---

## 8. Error Handling

* Fail early with clear messages

```python
if X.ndim != 2:
    raise ValueError("X must be a 2D array")
```

* Do not silently ignore errors

---

## 9. Consistency

* Follow the same patterns across the codebase
* Do not mix styles or conventions
* Consistency is more important than perfection

---

## 10. Before Committing

* Code is readable and well-named
* Types are specified
* Docstrings are present
* No obvious TODOs left unresolved
* Code passes formatter and linter

---
