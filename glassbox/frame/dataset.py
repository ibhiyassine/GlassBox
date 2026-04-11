from typing import List, Tuple

import numpy as np


class Dataset:
    """
    Container for data matrices with multiple helper functions.

    Attributes
    ----------
    data : np.ndarray
        Data arranged as a 2D matrix. To access columns, take the transpose.
    columns : List[str]
        Names of the columns stored in a list.
    shape : Tuple[int, int]
        Shape of the dataset (# of rows, # of columns).
    """

    __slots__ = ["_data", "_columns"]

    # ─[ init function ]────────────────────────────────────────────────────
    def __init__(self, data: np.ndarray, columns: List[str]):
        """
        Parameters
        ----------
        data: np.ndarray
            Data arranged as a 2D matrix - (n_rows, n_cols)
        columns: List[str]
            column names - must match data.shape[1]
        """
        if data.ndim != 2:
            raise ValueError(f"data must be a 2D matrix, got shape {data.ndim}")
        if len(columns) != data.shape[1]:
            raise ValueError(
                f"Number of columns ({len(columns)}) does not match"
                f"data width ({data.shape[1]})"
            )
        self._data = data
        self._columns = columns

    # ─[ Properties ]───────────────────────────────────────────────────────
    @property
    def shape(self) -> Tuple[int, int]:
        return self._data.shape

    @property
    def columns(self) -> List[str]:
        return self._columns

    @property
    def data(self) -> np.ndarray:
        return self._data

    # ─[ Private helpers ]──────────────────────────────────────────────────
    def _get_column_index(self, name: str) -> int:
        """
        Map a column name to its integer index.

        Parameters
        ----------
        name : str
            Name of the column.

        Returns
        -------
        int
            Index position of the column.

        Raises
        ------
        KeyError
            If the column name does not match
        """
        try:
            return self._columns.index(name)
        except ValueError:
            raise KeyError(f"column '{name}' not found. Available: {self.columns}")

    def _normalize_names(self, names: str | List[str]) -> List[str]:
        """
        Coerce a single name or a list of names into a list

        Parameters
        ----------
        names: str | List[str]
            The names to be grouped into a list

        Returns
        -------
        List[str]
            list of names
        """
        return [names] if isinstance(names, str) else list(names)

    # ─[ Public API ]───────────────────────────────────────────────────────
    def get_columns(self, names: str | List[str]) -> "Dataset":
        """
        Retrieve data for specific columns by name.

        Parameters
        ----------
        names : str | List[str]
            A single column name or a list of column names.

        Returns
        -------
        np.ndarray
            Array slice representing the requested columns.
        """
        names = self._normalize_names(names)
        indices = [self._get_column_index(name) for name in names]
        return Dataset(self._data[:, indices].copy(), names)

    def get_rows(self, indices: np.ndarray) -> "Dataset":
        """
        Get specific rows based on indices and return as a new dataset.

        Parameters
        ----------
        indices : np.ndarray
            Integer array of row coordinates.

        Returns
        -------
        Dataset
            A new Dataset instance containing the selected rows.
        """
        return Dataset(self._data[indices].copy(), self._columns)

    def update_column(self, name: str, new_data: np.ndarray) -> None:
        """
        Update the array content for an existing column.

        Parameters
        ----------
        name : str
            Target column to update.
        new_data : np.ndarray
            Array values to overwrite the column.

        Returns
        -------
        None
        """
        idx = self._get_column_index(name)
        self._data[:, idx] = new_data.ravel()

    def drop_columns(self, names: str | List[str]) -> None:
        """
        Remove columns by name from the dataset.

        Parameters
        ----------
        names : str | List[str]
            Target column or list of columns to remove.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            if one of the columns to drop doesn't exist in the dataset
        """
        to_drop = set(self._normalize_names(names))
        keep_mask = [i for i, col in enumerate(self._columns) if col not in to_drop]
        missing = to_drop - set(self._columns)
        if missing:
            raise KeyError(f"columns not found: {missing}")
        self._data = self._data[:, keep_mask]
        self._columns = [self._columns[i] for i in keep_mask]

    def add_columns(self, new_dataset: "Dataset") -> None:
        """
        Add new columns alongside the dataset arrays.

        Parameters
        ----------
        new_data: Dataset
            New data to append

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If column to add already exists
        """
        duplicates = [n for n in new_dataset.columns if n in self._columns]
        if duplicates:
            raise ValueError(f"Columns already exist: {duplicates}")

        if new_dataset.shape[0] != self._data.shape[0]:
            raise ValueError(
                f"Row count mismatch: dataset has {self._data.shape[0]} rows, "
                f"new data has {new_dataset.shape[0]}"
            )

        self._data = np.hstack([self._data, new_dataset.data])
        self._columns.extend(new_dataset.columns)

    # ─[ Representation of class ]──────────────────────────────────────────
    def __repr__(self) -> str:
        return f"Dataset(shape={self.shape}, " f"columns={self._columns})"
