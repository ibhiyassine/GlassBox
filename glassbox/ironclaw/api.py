import json
from typing import Dict, Any, List

from glassbox.frame import read_csv
from glassbox.inspector import DataAuditor
from glassbox.ironclaw.state import memory

def inspect_data_api(csv_filename: str) -> str:
    """
    Called by IronClaw agent initially to load data and generate EDA stats.
    """
    try:
        # Load the raw file securely in the Pyodide filesystem
        dataset = read_csv(csv_filename)
        
        # Retain dataset globally for cleaner/trainer
        memory.set("dataset", dataset)
        
        # Run deep inspection utilizing GlassBox module
        auditor = DataAuditor()
        report = auditor.run_audit(dataset)
        
        return json.dumps({
            "status": "success",
            "message": f"Successfully loaded {csv_filename} into memory.",
            # Return serialized stats to the LLM orchestrator
            "eda_report": report.to_dict() if hasattr(report, "to_dict") else str(report)
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

def clean_data_api(
    imputation_strategy: Dict[str, str],
    outlier_handling: Dict[str, str],
    encoding_strategy: Dict[str, str],
    scale: str
) -> str:
    """
    Safely mutates the data applying exactly the logic requested by the LLM schemas.
    """
    try:
        dataset = memory.verify_dataset()
        
        import numpy as np
        from glassbox.cleaner import (
            SimpleImputer, ImputationStrategy, OutlierCapper,
            MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
        )
        from glassbox.frame.dataset import Dataset
        
        # 1. Imputation
        for col, strategy in imputation_strategy.items():
            if col in dataset.columns:
                try:
                    strat = ImputationStrategy(strategy.upper())
                except ValueError:
                    strat = ImputationStrategy.MEAN
                imputer = SimpleImputer(strategy=strat)
                col_data = dataset.get_columns([col]).data
                new_data = imputer.fit_transform(col_data)
                dataset.update_column(col, new_data)

        # 2. Outliers
        for col, method in outlier_handling.items():
            if col in dataset.columns and method.lower() == "clip":
                capper = OutlierCapper()
                col_data = dataset.get_columns([col]).data
                new_data = capper.fit_transform(col_data)
                dataset.update_column(col, new_data)
                
        # 3. Encoding
        to_drop = []
        for col, strat in encoding_strategy.items():
            if col not in dataset.columns:
                continue
            col_data = dataset.get_columns([col]).data
            if strat.lower() == "label":
                enc = LabelEncoder()
                new_data = enc.fit_transform(col_data)
                dataset.update_column(col, new_data)
            elif strat.lower() in ("onehot", "one-hot"):
                enc = OneHotEncoder()
                new_data = enc.fit_transform(col_data)
                cats = enc._categories.get(0, [])
                new_cols = [f"{col}_{cat}" for cat in cats]
                new_ds = Dataset(new_data, new_cols)
                dataset.add_columns(new_ds)
                to_drop.append(col)
                
        if to_drop:
            dataset.drop_columns(to_drop)

        # 4. Scaling
        if scale:
            scaler = None
            if scale.lower() == "minmax":
                scaler = MinMaxScaler()
            elif scale.lower() == "standard":
                scaler = StandardScaler()
                
            if scaler:
                numeric_cols = []
                for c in dataset.columns:
                    col_data = dataset.get_columns([c]).data[:, 0]
                    if np.issubdtype(col_data.dtype, np.number):
                        numeric_cols.append(c)
                    else:
                        try:
                            col_data.astype(float)
                            numeric_cols.append(c)
                        except (ValueError, TypeError):
                            pass
                            
                if numeric_cols:
                    col_data = dataset.get_columns(numeric_cols).data.astype(float)
                    new_data = scaler.fit_transform(col_data)
                    for i, c in enumerate(numeric_cols):
                        dataset.update_column(c, new_data[:, i:i+1])
        
        return json.dumps({
            "status": "success",
            "message": "Data cleaned and updated in memory state.",
            "configurations_applied": {
                "imputation": list(imputation_strategy.keys()),
                "outliers": list(outlier_handling.keys()),
                "encoded": list(encoding_strategy.keys()),
                "scaling": scale
            }
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

def train_and_tune_api(
    target_column: str,
    models: List[str],
    metric: str,
    metric_direction: str
) -> str:
    """
    Hunts for the best ML model on the cleaned data using specific algorithms.
    """
    try:
        dataset = memory.verify_dataset()
        
        # Prepare evaluation inputs
        feature_cols = [c for c in dataset.columns if c != target_column]
        X = dataset.get_columns(feature_cols).data.astype(float)
        y = dataset.get_columns(target_column).data[:, 0].astype(float)
        
        # Map scoring function
        import glassbox.metrics as smetrics
        import glassbox.models as gmodels
        from glassbox.orchestrator import GridSearchCV, KFoldSplitter
        
        metric_func = getattr(smetrics, f"{metric}_score", None)
        if metric_func is None:
            # Try plain metric if "_score" suffix was redundant
            metric_func = getattr(smetrics, metric, smetrics.accuracy_score)
            
        cv = KFoldSplitter(n_splits=3)
        
        model_space = {
            "knn": (gmodels.KNeighborsClassifier(), {"n_neighbors": [3, 5, 7]}),
            "decision_tree": (gmodels.DecisionTreeClassifier(), {"max_depth": [3, 5, 10]}),
            "random_forest": (gmodels.RandomForestClassifier(), {"n_estimators": [10, 30]})
        }
        
        best_score = float("-inf") if metric_direction == "max" else float("inf")
        best_model_name = None
        best_params = None
        
        for m_name in models:
            if m_name not in model_space:
                continue
                
            estimator, p_grid = model_space[m_name]
            
            search = GridSearchCV(
                estimator=estimator,
                param_space=p_grid,
                cv_engine=cv,
                scoring_func=metric_func
            )
            
            # Since some models might fail if data is not cleanly processed, use simple try
            search.fit(X, y)
            
            score = search.best_score_
            
            if metric_direction == "max":
                better = score > best_score
            else:
                better = score < best_score
                
            if better:
                best_score = score
                best_model_name = m_name
                best_params = search.best_params_
        
        return json.dumps({
            "status": "success",
            "message": f"Trained {len(models)} models aiming to {metric_direction} {metric}.",
            "best_model": {
                "name": best_model_name,
                "score": best_score,
                "params": best_params
            }
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})
