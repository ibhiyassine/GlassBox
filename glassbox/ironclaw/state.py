"""
In-memory persistence for stateless WASM environments.
Since LLM tool calls execute independently in the Pyodide loop, we must retain 
robust pointers to the active GlassBox Dataset between Inspector, Cleaner, and Trainer hits.
"""
from typing import Any, Dict

class StateManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StateManager, cls).__new__(cls)
            cls.storage: Dict[str, Any] = {}
        return cls._instance

    def set(self, key: str, value: Any) -> None:
        self.storage[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.storage.get(key, default)

    def verify_dataset(self) -> Any:
        """
        Validates safety rule enforcing LLM calls InspectDataAPI before attempting
        to clean or train on nil references.
        """
        dataset = self.get("dataset")
        if dataset is None:
            raise ValueError("No active dataset. The agent must run InspectDataAPI first.")
        return dataset

# Global pointer initialized into memory scope for Pyodide
memory = StateManager()
