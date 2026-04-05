from typing import List

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
    raise NotImplementedError


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
    raise NotImplementedError
