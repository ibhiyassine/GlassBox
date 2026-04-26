# IronClaw: Agentic API

IronClaw is the agentic interface for GlassBox, designed specifically for **WASM-based environments** (like Pyodide). It provides a high-level, stateless API that allows LLM agents to orchestrate complex AutoML pipelines.

## Key Features

- **Stateless Orchestration**: Enables independent tool calls by managing an in-memory state.
- **Agent-Ready Schemas**: Functions are designed to be easily invoked by LLMs with clear input/output structures.
- **WASM Optimized**: Lightweight and compatible with browser-based Python execution.

## Architecture

The following diagram illustrates how IronClaw interacts with the core GlassBox library and the Agent:

```kroki-plantuml
@from_file:puml/ironclaw_arch.puml
```

## Main API Functions

### `inspect_data_api(csv_filename)`
Loads a CSV file into memory and generates a comprehensive EDA report.

### `clean_data_api(...)`
Applies a suite of cleaning operations (imputation, encoding, scaling, etc.) based on agent-defined strategies.

### `train_and_tune_api(...)`
Orchestrates model selection and hyperparameter optimization to find the best performing model for a target column.

## State Management

Since web-based agents often call tools in a loop where Python state might be reset or handled carefully, IronClaw uses a global `StateManager` to retain pointers to the active dataset.

```python
from glassbox.ironclaw.state import memory

# Inspect first to load into memory
inspect_data_api("data.csv")

# Subsequent calls automatically find the dataset
clean_data_api(...)
```
