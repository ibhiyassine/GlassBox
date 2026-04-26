# Metrics

The `glassbox.metrics` module provides functions to evaluate the performance of **classification** and **regression** models. All functions take true labels (`y_true`) and predicted labels (`y_pred`) as inputs.

```kroki-plantuml
@from_file:puml/metrics.puml
```

---

## Classification Metrics

Evaluate models predicting discrete class labels.

### accuracy_score

Calculates the proportion of correctly classified samples.

```python
from glassbox.metrics import accuracy_score

acc = accuracy_score(y_true, y_pred)
```

### precision_score

Calculates the ratio of true positives to total predicted positives.

```python
from glassbox.metrics import precision_score

precision = precision_score(y_true, y_pred)
```

### recall_score

Calculates the ratio of true positives to total actual positives.

```python
from glassbox.metrics import recall_score

recall = recall_score(y_true, y_pred)
```

### f1_score

Calculates the harmonic mean of precision and recall.

```python
from glassbox.metrics import f1_score

f1 = f1_score(y_true, y_pred)
```

### confusion_matrix

Generates a confusion matrix showing true vs predicted classes.

```python
from glassbox.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)
```

---

## Regression Metrics

Evaluate models predicting continuous numeric targets.

### mean_absolute_error (MAE)

Calculates the average absolute difference between predicted and actual values.

```python
from glassbox.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)
```

### mean_squared_error (MSE)

Calculates the average squared difference between predicted and actual values.

```python
from glassbox.metrics import mean_squared_error

mse = mean_squared_error(y_true, y_pred)
```

### r2_score

Calculates the coefficient of determination (R²), indicating the proportion of variance explained by the model.

```python
from glassbox.metrics import r2_score

r2 = r2_score(y_true, y_pred)
```

---

## API Reference

::: glassbox.metrics
    options:
      show_root_heading: false
      members:
        - accuracy_score
        - precision_score
        - recall_score
        - f1_score
        - confusion_matrix
        - mean_absolute_error
        - mean_squared_error
        - r2_score
