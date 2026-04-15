# Inspector

The `glassbox.inspector` module performs a **non-destructive audit** of raw data, producing a comprehensive `EDAReport` with feature typing, statistics, outlier detection, and association analysis.

---

## Running a Full Audit

The `DataAuditor` orchestrates the entire EDA pipeline in a single call:

```python
from glassbox.frame import read_csv
from glassbox.inspector import DataAuditor

ds = read_csv("data.csv")
auditor = DataAuditor()
report = auditor.run_audit(ds)
```

The returned `EDAReport` contains five sections:

| Field | Type | Description |
|---|---|---|
| `feature_types` | `Dict[str, FeatureType]` | Auto-detected type per column. |
| `missing_values` | `Dict[str, MissingInfo]` | Missing count & percentage. |
| `outliers_info` | `Dict[str, OutlierInfo]` | IQR-based outlier bounds & counts. |
| `summary_stats` | `Dict[str, NumericStats \| CategoricalStats]` | Descriptive statistics. |
| `collinearity_map` | `List[CollinearityPair]` | Pairwise associations. |

---

## Auto-Typing

The `AutoTyper` classifies each column into one of four logical types:

| FeatureType | Criteria |
|---|---|
| `BOOLEAN` | Exactly 2 unique values. |
| `ORDINAL` | Integer values with cardinality < 20. |
| `NUMERICAL` | Continuous float values. |
| `NOMINAL` | Non-numeric (string/categorical). |

```python
from glassbox.inspector import AutoTyper

typer = AutoTyper()
types = typer.infer_types(ds)
# {'Age': NUMERICAL, 'Gender': NOMINAL, 'Passed': BOOLEAN, ...}
```

---

## Statistical Profiling

`StatProfiler` computes summary statistics split by feature type:

**Numeric features** → `NumericStats`:

- Mean, Median, Standard Deviation, Skewness, Kurtosis

**Categorical features** → `CategoricalStats`:

- Mode, Cardinality (number of unique values)

```python
from glassbox.inspector import StatProfiler

profiler = StatProfiler()

num_stats = profiler.calculate_numeric_stats(ds, ["Age", "Score"])
cat_stats = profiler.calculate_categorical_stats(ds, ["Gender"])
```

---

## Outlier Detection

`OutlierDetector` uses the **Interquartile Range (IQR)** method:

- **Lower bound**: Q1 − 1.5 × IQR
- **Upper bound**: Q3 + 1.5 × IQR

```python
from glassbox.inspector import OutlierDetector

detector = OutlierDetector()
outliers = detector.flag_outliers(ds, ["Age", "Score"])

print(outliers["Age"])
# OutlierInfo(count=12, lower_bound=5.0, upper_bound=65.0)
```

---

## Association Analysis

`AssociationAnalyzer` computes pairwise associations:

- **Numeric ↔ Numeric**: Pearson correlation coefficient
- **Categorical ↔ Categorical**: Cramér's V statistic

```python
from glassbox.inspector import AssociationAnalyzer

analyzer = AssociationAnalyzer()
pairs = analyzer.build_associations(
    ds,
    num_cols=["Age", "Score"],
    cat_cols=["Gender", "Region"],
)

for pair in pairs:
    print(f"{pair.feature_a} ↔ {pair.feature_b}: "
          f"{pair.score:.3f} ({pair.metric})")
```

---

## Serialization

The full report can be serialized to JSON:

```python
json_str = report.to_json()
```

??? note
    `NaN` values are serialized as `null` and `FeatureType` enums are serialized by name (e.g., `"NUMERICAL"`).

---

## API Reference

::: glassbox.inspector
    options:
      show_root_heading: false
      members:
        - DataAuditor
        - AutoTyper
        - StatProfiler
        - AssociationAnalyzer
        - OutlierDetector
        - EDAReport
        - OutlierInfo
