# GlassBox 🔍

[![PyPI version](https://img.shields.io/pypi/v/glassbox-ahii.svg)](https://pypi.org/project/glassbox-ahii/)
[![Documentation Status](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://ibhiyassine.github.io/GlassBox/)
[![Frontend](https://img.shields.io/badge/IronClaw-Live%20Demo-brightgreen)](https://glassbox-automl.netlify.app)
[![Build Status](https://img.shields.io/github/actions/workflow/status/ibhiyassine/GlassBox/docs.yml?branch=main&label=docs)](https://github.com/ibhiyassine/GlassBox/actions)

**GlassBox** is a professional, white-box AutoML library for Python designed to make machine learning interpretable, transparent, and accessible. Unlike "black-box" models, GlassBox prioritizes clarity, offering a suite of glass-box models (Decision Trees, KNN, Linear Models) and a comprehensive orchestration layer for automated data cleaning, inspection, and tuning.

## 🚀 Key Features

- **White-Box Models**: Fully interpretable implementations of standard algorithms (Trees, Neighbors, Ensembles).
- **IronClaw Agent**: An agentic WASM-powered interface for interactive, chat-based data analysis.
- **Automated Orchestration**: One-line training pipelines with built-in hyperparameter tuning.
- **Robust Cleaning**: Intelligent data transformers for dealing with missing values, categorical encoding, and scaling.
- **Rich Documentation**: Comprehensive guides with architectural diagrams.

## 📦 Installation

```bash
pip install glassbox-ahii
```

## 🛠️ Quick Start

```python
from glassbox import Orchestrator
from glassbox.frame import Dataset
import pandas as pd

# Load your data
df = pd.read_csv("data.csv")
data = Dataset(df, target="target_column")

# Train and tune with one line
orchestrator = Orchestrator()
best_model = orchestrator.train_and_tune(data)

# Predict
predictions = best_model.predict(df.drop("target_column", axis=1))
```

## 🤖 IronClaw (Web Agent)

Experience the Power of GlassBox in your browser! **IronClaw** is our agentic web interface that uses Pyodide (Python in WASM) to run GlassBox directly client-side. No data ever leaves your browser.

👉 **[Try IronClaw Live](https://glassbox-automl.netlify.app)**

## 📚 Documentation

Explore our full documentation, including architectural diagrams and API references:
**[https://ibhiyassine.github.io/GlassBox/](https://ibhiyassine.github.io/GlassBox/)**

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
