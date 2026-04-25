"""Generate the code reference pages with richer formatting."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
src = root / "glassbox"

# Human-readable module descriptions
MODULE_DESCRIPTIONS = {
    "glassbox": "Top-level package for the GlassBox library.",
    "glassbox.core": "Core mathematical primitives used across all modules.",
    "glassbox.core.math": "Low-level statistical functions, distance metrics, and tree split utilities — all implemented with NumPy.",
    "glassbox.frame": "Data container and CSV I/O utilities.",
    "glassbox.frame.dataset": "The `Dataset` class — a lightweight, named-column wrapper around a 2-D NumPy array.",
    "glassbox.frame.io": "Functions for reading and writing CSV files.",
    "glassbox.inspector": "Non-destructive Exploratory Data Analysis (EDA) toolkit.",
    "glassbox.inspector.auditor": "The `DataAuditor` — orchestrates the full EDA pipeline.",
    "glassbox.inspector.auto_typer": "Automatic feature type inference (Numerical, Nominal, Ordinal, Boolean).",
    "glassbox.inspector.outliers": "IQR-based outlier detection for numeric columns.",
    "glassbox.inspector.report": "Data classes for the EDA report: `EDAReport`, `OutlierInfo`, `MissingInfo`, `NumericStats`, `CategoricalStats`.",
    "glassbox.inspector.statistics": "Statistical profiling and pairwise association analysis (Pearson, Cramér's V).",
    "glassbox.cleaner": "Scikit-learn-style data cleaning transformers.",
    "glassbox.cleaner._base": "Abstract `BaseTransformer` defining the fit / transform interface.",
    "glassbox.cleaner.imputers": "`SimpleImputer` — replaces missing values using mean, median, mode, or a constant.",
    "glassbox.cleaner.outliers": "`OutlierCapper` — clips values outside IQR bounds.",
    "glassbox.cleaner.scalers": "`StandardScaler` and `MinMaxScaler` for feature normalization.",
    "glassbox.cleaner.encoders": "`OneHotEncoder` and `LabelEncoder` for categorical feature encoding.",
    "glassbox.models": "Machine learning models for classification and regression.",
    "glassbox.models._base": "Abstract `BaseModel` defining the fit / predict interface.",
    "glassbox.models.trees": "Decision tree models (CART).",
    "glassbox.models.trees._base": "Abstract `BaseTree` with recursive tree building and traversal logic.",
    "glassbox.models.trees.classifier": "`DecisionTreeClassifier` — splits on Gini impurity.",
    "glassbox.models.trees.regressor": "`DecisionTreeRegressor` — splits on variance reduction.",
    "glassbox.models.ensemble": "Random Forest ensemble models.",
    "glassbox.models.ensemble._base": "Abstract `BaseRandomForest` with bootstrap sampling and feature subsets.",
    "glassbox.models.ensemble.classifier": "`RandomForestClassifier` — majority-vote aggregation.",
    "glassbox.models.ensemble.regressor": "`RandomForestRegressor` — mean aggregation.",
    "glassbox.models.neighbors": "K-Nearest Neighbors models.",
    "glassbox.models.neighbors._enums": "`DistanceMetric` and `SearchAlgorithm` enums.",
    "glassbox.models.neighbors._knn": "`KNeighborsClassifier` and `KNeighborsRegressor`.",
    "glassbox.models.neighbors.index": "Spatial index implementations for KNN search.",
    "glassbox.models.neighbors.index._base": "Abstract `BaseIndex` interface for nearest-neighbor lookup.",
    "glassbox.models.neighbors.index._brute": "`BruteForceIndex` — exhaustive distance search.",
    "glassbox.models.neighbors.index._kdtree": "`KDTreeIndex` — space-partitioning tree for efficient nearest-neighbor queries.",
    "glassbox.models.gaussian_nb": "Gaussian Naive Bayes models.",
    "glassbox.models.gaussian_nb._base": "Abstract base class for GaussianNB.",
    "glassbox.models.gaussian_nb.gaussian_nb": "`GaussianNB` classifier.",
    "glassbox.models.linear_model": "Linear regression and classification models.",
    "glassbox.models.linear_model._base": "Abstract `BaseLinearModel`.",
    "glassbox.models.linear_model._enums": "`LearningSchedule` enum.",
    "glassbox.models.linear_model.linear": "`LinearRegression` model.",
    "glassbox.models.linear_model.logistic": "`LogisticRegression` model.",
    "glassbox.metrics": "Evaluation metrics for classification and regression.",
    "glassbox.metrics.classification": "Classification metrics (accuracy, precision, recall, f1_score).",
    "glassbox.metrics.regression": "Regression metrics (MSE, MAE, R2).",
    "glassbox.orchestrator": "Model selection and hyperparameter search tools.",
    "glassbox.orchestrator.base_search": "Abstract `BaseSearch` for grid/randomized search.",
    "glassbox.orchestrator.base_splitter": "Abstract `BaseSplitter` for cross-validation.",
    "glassbox.orchestrator.grid_search": "`GridSearchCV` for exhaustive search.",
    "glassbox.orchestrator.randomized_search": "`RandomizedSearchCV` for randomized search.",
    "glassbox.orchestrator.splitters": "Cross-validation generators (`KFoldSplitter`, `StratifiedKFoldSplitter`).",
}

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(root).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue
    elif parts[-1].startswith("__"):
        continue

    nav[parts] = doc_path.as_posix()

    ident = ".".join(parts)
    description = MODULE_DESCRIPTIONS.get(ident, "")

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        # Title
        fd.write(f"---\ntitle: {parts[-1]}\n---\n\n")
        # Heading
        fd.write(f"# `{ident}`\n\n")
        # Description
        if description:
            fd.write(f"{description}\n\n")
        fd.write("---\n\n")
        # API docs directive
        fd.write(f"::: {ident}\n")
        fd.write("    options:\n")
        fd.write("      show_root_heading: false\n")
        fd.write("      show_source: true\n")
        fd.write("      heading_level: 2\n")
        fd.write("      members_order: source\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
