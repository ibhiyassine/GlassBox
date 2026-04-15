# GlassBox

**A transparent, white-box AutoML library built from scratch with NumPy.**

GlassBox is a lightweight AutoML library designed for auditability and clarity. Every algorithm is implemented from first principles using only NumPy — no hidden black-box dependencies.

---

## Features

<div class="grid cards" markdown>

-   :material-table-edit:{ .lg .middle } **Frame**

    ---

    A minimal `Dataset` container with CSV I/O, column selection, row slicing, and in-place mutations — all backed by NumPy arrays.

    [:octicons-arrow-right-24: Frame Guide](user-guide/frame.md)

-   :material-magnify:{ .lg .middle } **Inspector**

    ---

    Non-destructive Exploratory Data Analysis: statistical profiling, auto-typing, outlier detection, and Pearson / Cramér's V associations.

    [:octicons-arrow-right-24: Inspector Guide](user-guide/inspector.md)

-   :material-broom:{ .lg .middle } **Cleaner**

    ---

    A scikit-learn-style `fit` / `transform` pipeline for imputation, outlier capping, scaling, and encoding.

    [:octicons-arrow-right-24: Cleaner Guide](user-guide/cleaner.md)

-   :material-brain:{ .lg .middle } **Models**

    ---

    Decision Trees, Random Forests, and K-Nearest Neighbors — for both classification and regression.

    [:octicons-arrow-right-24: Models Guide](user-guide/models.md)

</div>

---

## Quick Install

```bash
pip install .          # runtime only (numpy)
pip install .[dev]     # with dev tools (mkdocs, ruff, black)
```

---

## Philosophy

| Principle | Meaning |
|---|---|
| **Clarity over cleverness** | Every line should be readable. |
| **Explicit over implicit** | No hidden transformations. |
| **NumPy is the engine** | Zero heavy dependencies. |
| **Auditability first** | Built for white-box inspection. |
