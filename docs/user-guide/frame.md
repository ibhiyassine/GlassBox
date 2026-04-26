# Frame

The `glassbox.frame` module provides a lightweight data container and CSV I/O utilities. All data is stored internally as `np.ndarray` — no pandas dependency.

---

## Loading Data

Use `read_csv` to load a CSV file into a `Dataset`:

```python
from glassbox.frame import read_csv

ds = read_csv("students.csv")
print(ds)
# Dataset(shape=(1000, 12), columns=['Age', 'Gender', 'Score', ...])
```

??? info "Type inference"
    Columns that can be fully cast to `float` are stored as `float64`.
    Mixed or string columns remain as `object` dtype.
    Empty cells and `"NA"` values are converted to `np.nan`.

---

## The Dataset Class

```kroki-plantuml
@from_file:puml/frame.puml
```

`Dataset` wraps a 2-D NumPy array with named columns.

### Properties

| Property | Type | Description |
|---|---|---|
| `data` | `np.ndarray` | The underlying 2-D array. |
| `columns` | `List[str]` | Column names. |
| `shape` | `Tuple[int, int]` | `(n_rows, n_cols)`. |

### Selecting Columns

```python
# Single column → Dataset with 1 column
ages = ds.get_columns("Age")

# Multiple columns → Dataset subset
subset = ds.get_columns(["Age", "Score"])
```

### Selecting Rows

```python
import numpy as np

indices = np.array([0, 5, 10])
sample = ds.get_rows(indices)
```

### Modifying Data

```python
# Update an existing column
ds.update_column("Score", new_scores)

# Drop columns
ds.drop_columns(["Unused_1", "Unused_2"])

# Add new columns from another Dataset
ds.add_columns(extra_ds)
```

---

## Saving Data

```python
from glassbox.frame.io import to_csv

to_csv(ds, columns=["Age", "Score"], filepath="output.csv")
```

??? tip
    `to_csv` automatically formats whole-number floats without a decimal point and properly escapes commas and quotes in string values.

---

## API Reference

::: glassbox.frame
    options:
      show_root_heading: false
      members:
        - Dataset
        - read_csv
