from typing import List

import numpy as np

from .dataset import Dataset


def read_csv(filepath: str) -> Dataset:
    """
    Load a CSV file into a Dataset.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    Dataset
        Loaded dataset object.
    """
    with open(filepath, newline="", encoding="utf-8") as data_source:
        entries = [entry.rstrip("\n") for entry in data_source if entry.strip()]

    if not entries:
        raise ValueError(f"CSV file is empty: {filepath}")

    # Parse header
    columns = [col.strip().strip('"') for col in entries[0].split(",")]

    # Parse rows
    rows = []
    for line in entries[1:]:
        cells = [cell.strip().strip('"') for cell in line.split(",")]
        rows.append(cells)

    if not rows:
        # Header-only file — return empty dataset with float64 dtype
        data = np.empty((0, len(columns)), dtype=object)
        return Dataset(data, columns)

    # Build a column-at-a-time array, trying float conversion per column
    n_rows = len(rows)
    n_cols = len(columns)
    data = np.empty((n_rows, n_cols), dtype=object)

    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            data[r, c] = cell

    # Try to cast each column to float; leave as object if it fails
    float_data = np.empty((n_rows, n_cols), dtype=float)
    col_is_float = np.ones(n_cols, dtype=bool)

    for c in range(n_cols):
        try:
            float_data[:, c] = data[:, c].astype(float)
        except (ValueError, TypeError):
            col_is_float[c] = False

    if col_is_float.all():
        return Dataset(float_data, columns)

    # Mixed types — keep object array as-is; promote pure-float cols
    for c in range(n_cols):
        if col_is_float[c]:
            data[:, c] = float_data[:, c]

    return Dataset(data, columns)


def to_csv(data: Dataset, columns: List[str], filepath: str) -> None:
    """
    Save a Dataset to a CSV file on disk.

    Parameters
    ----------
    data : Dataset
        The dataset to save.
    columns : List[str]
        List of column names to save.
    filepath : str
        Destination path for the CSV file.

    Returns
    -------
    None
    """
    subset = data.get_columns(columns)  # shape (n_rows, len(columns))

    with open(filepath, "w", encoding="utf-8", newline="") as fh:
        # Header
        fh.write(",".join(columns) + "\n")

        # Rows
        for row in subset.data:
            fh.write(",".join(_format_cell(v) for v in row) + "\n")


# ─[ Internal Helper ]──────────────────────────────────────────────────
def _format_cell(value) -> str:
    """
    Render a single cell as a CSV-safe string.

    * Floats that are whole numbers are written without a decimal point.
    * Strings containing commas or quotes are quoted and escaped.
    """
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else repr(value)
    text = str(value)
    if "," in text or '"' in text:
        text = '"' + text.replace('"', '""') + '"'
    return text
