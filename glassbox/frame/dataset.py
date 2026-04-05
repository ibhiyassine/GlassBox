from typing import List

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

    __slots__ = ["_data", "_columns", "shape"]

    @property
    def columns(self) -> List[str]:
        """
        Get the list of column names.

        Returns
        -------
        List[str]
            Column names.
        """
        return self._columns

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
        """
        raise NotImplementedError

    def get_columns(self, names: str | List[str]) -> np.ndarray:
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
        raise NotImplementedError

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
        raise NotImplementedError

    def update_column(self, name: str, new_data: np.ndarray):
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
        raise NotImplementedError

    def drop_columns(self, names: str | List[str]):
        """
        Remove columns by name from the dataset.

        Parameters
        ----------
        names : str | List[str]
            Target column or list of columns to remove.

        Returns
        -------
        None
        """
        raise NotImplementedError

    def add_columns(self, names: str | List[str], data: np.ndarray):
        """
        Add new columns alongside the dataset arrays.

        Parameters
        ----------
        names : str | List[str]
            Name or names of the new columns.
        data : np.ndarray
            Values for the new columns.

        Returns
        -------
        None
        """
        raise NotImplementedError
