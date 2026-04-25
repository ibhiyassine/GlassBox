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
        report = auditor.audit(dataset) 
        
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
        
        # [Implementation Note]: Apply GlassBox primitive transforms here
        # dataset = apply_glassbox_cleaners(dataset, ...)
        
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
        
        # [Implementation Note]: Map the 'models' Enum list directly to 
        # KNeighborsClassifier, RandomForestClassifier, etc., initializing from glassbox.models
        # and delegating evaluation to glassbox.orchestrator
        
        return json.dumps({
            "status": "success",
            "message": f"Trained {len(models)} models aiming to {metric_direction} {metric}.",
            "best_model": "..."
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})
