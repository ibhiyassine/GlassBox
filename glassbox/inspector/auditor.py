import numpy as np

from glassbox.frame.dataset import Dataset
from glassbox.inspector.auto_typer import AutoTyper
from glassbox.inspector.outliers import OutlierDetector
from glassbox.inspector.report import EDAReport, FeatureType, MissingInfo
from glassbox.inspector.statistics import AssociationAnalyzer, StatProfiler


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
        auto_typer = AutoTyper()
        outlier_detector = OutlierDetector()
        stat_profiler = StatProfiler()
        association_analyzer = AssociationAnalyzer()

        feature_types = auto_typer.infer_types(data)
        n_samples = data.shape[0]

        missing_values = {}
        for col_name in data.columns:
            col_data = data.get_columns(col_name).data[:, 0]
            if np.issubdtype(col_data.dtype, np.number):
                missing = int(np.isnan(col_data).sum())
            else:
                missing = 0
                for v in col_data:
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        missing += 1
            missing_values[col_name] = MissingInfo(count=missing, percentage=missing / n_samples)

        numeric_cols = [c for c, t in feature_types.items() if t == FeatureType.NUMERICAL]
        categorical_cols = [c for c, t in feature_types.items() if t in (FeatureType.NOMINAL, FeatureType.ORDINAL, FeatureType.BOOLEAN)]

        outliers_info = outlier_detector.flag_outliers(data, numeric_cols)
        num_stats = stat_profiler.calculate_numeric_stats(data, numeric_cols)
        cat_stats = stat_profiler.calculate_categorical_stats(data, categorical_cols)
        collinearity_map = association_analyzer.build_associations(data, numeric_cols, categorical_cols)

        summary_stats = {**num_stats, **cat_stats}

        return EDAReport(
            feature_types=feature_types,
            missing_values=missing_values,
            outliers_info=outliers_info,
            summary_stats=summary_stats,
            collinearity_map=collinearity_map
        )
