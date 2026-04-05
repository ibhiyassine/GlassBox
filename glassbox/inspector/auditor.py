from glassbox.frame.dataset import Dataset

from .report import EDAReport


class DataAuditor:
    """
    Orchestrates the EDA process to generate a complete report.
    """

    def run_audit(self, data: Dataset) -> EDAReport:
        """
        Perform a full audit on the dataset.

        Parameters
        ----------
        data : Dataset
            The dataset to audit.

        Returns
        -------
        EDAReport
            A comprehensive report containing EDA results.
        """
        raise NotImplementedError
