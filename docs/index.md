# GlassBox Documentation

Welcome to the documentation for **GlassBox**, a white-box AutoML library.

## Getting Started

GlassBox provides clear, interpretability-focused data inspection and modeling.

To start, check out our modules like `glassbox.frame` and `glassbox.inspector`.

### Inspector Module

The `glassbox.inspector` module performs a non-destructive audit of raw data to provide context.

Key capabilities:
- **Statistical Profiling**: Manual calculation of Mean, Median, Mode, Standard Deviation, Skewness, and Kurtosis.
- **Association Analysis**: Constructing a Pearson Correlation Matrix from scratch to identify collinearity.
- **Outlier Detection**: Interquartile Range (IQR) logic to flag possible data points falling beyond 1.5 × IQR.
- **Auto-Typing**: Smart automated logic prioritizing distinguishing between Numerical, Categorical (Nominal), and Boolean data types.
