import json

import numpy as np

from glassbox.cleaner import (
    ImputationStrategy,
    OutlierCapper,
    SimpleImputer,
    StandardScaler,
)
from glassbox.frame import Dataset, read_csv
from glassbox.inspector import DataAuditor
from glassbox.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from glassbox.models import (
    DecisionTreeClassifier,
    DistanceMetric,
    KNeighborsClassifier,
    LearningSchedule,
    LinearRegression,
    LogisticRegression,
    RandomForestClassifier,
    SearchAlgorithm,
)


def main():
    filepath = "datasets/Penguins/penguins.csv"
    print(f"Loading dataset from {filepath}...")
    dataset = read_csv(filepath)
    print(f"Loaded {dataset.shape[0]} rows and {dataset.shape[1]} columns.")

    auditor = DataAuditor()

    print("\n" + "=" * 50)
    print("EDA REPORT: BEFORE CLEANING")
    print("=" * 50)
    report_before = auditor.run_audit(dataset)
    print(json.dumps(json.loads(report_before.to_json()), indent=4))

    print("\n" + "-" * 50)
    print("Running Cleaner Modules Pipeline...")
    print("-" * 50)

    # penguins dataset indices:
    # 0: id, 1: species, 2: island, 3: bill_length_mm, 4: bill_depth_mm
    # 5: flipper_length_mm, 6: body_mass_g, 7: sex, 8: year
    num_idx = [0, 3, 4, 5, 6, 8]
    cat_idx = [1, 2, 7]

    cleaned_data_matrix = dataset.data.copy()
    num_data = np.array(cleaned_data_matrix[:, num_idx], dtype=float)

    print("-> Applying Imputation (MEAN)...")
    imputer = SimpleImputer(strategy=ImputationStrategy.MEAN)
    imputed_num = imputer.fit_transform(num_data)

    print("-> Applying Scaler (StandardScaler)...")
    scaler = StandardScaler()
    scaled_num = scaler.fit_transform(imputed_num)

    print("-> Applying Outlier Capper...")
    capper = OutlierCapper()
    capped_num = capper.fit_transform(scaled_num)

    # Impute categorical features
    cat_data = cleaned_data_matrix[:, cat_idx]
    print("-> Applying Imputation (MODE) for categoricals...")
    cat_imputer = SimpleImputer(strategy=ImputationStrategy.MODE)
    imputed_cat = cat_imputer.fit_transform(cat_data)

    cleaned_data_matrix[:, num_idx] = capped_num
    cleaned_data_matrix[:, cat_idx] = imputed_cat

    clean_dataset = Dataset(cleaned_data_matrix, dataset.columns)

    print("\n" + "=" * 50)
    print("EDA REPORT: AFTER CLEANING")
    print("=" * 50)
    report_after = auditor.run_audit(clean_dataset)
    print(json.dumps(json.loads(report_after.to_json()), indent=4))

    print("\n" + "=" * 50)
    print("KNN CLASSIFIER TEST (Train/Test Split)")
    print("=" * 50)

    X = clean_dataset.data[:, num_idx]
    y = clean_dataset.data[:, cat_idx[0]]  # Species column

    # Shuffle and split (80/20)
    np.random.seed(42)  # For reproducibility
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    split_idx = int(0.8 * n_samples)

    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"Training on {len(train_idx)} samples. Testing on {len(test_idx)} samples.\n")

    knn_kd = KNeighborsClassifier(
        k=5, metric=DistanceMetric.EUCLIDEAN, algorithm=SearchAlgorithm.KD_TREE
    )
    print("Training KNN (KD-Tree)...")
    knn_kd.fit(X_train, y_train)
    preds = knn_kd.predict(X_test)

    accuracy = accuracy_score(y_true=y_test, y_pred=preds)
    print(f"KD-Tree Accuracy: {accuracy * 100:.2f}%\n")

    print("Sample of 15 Predictions (KD-Tree vs Actual):")
    for i in range(min(15, len(y_test))):
        match = "✓" if preds[i] == y_test[i] else "✗"
        print(f"[{match}] Predicted: {preds[i]:>10} | Actual: {y_test[i]:>10}")

    print("\n" + "-" * 50)

    knn_br = KNeighborsClassifier(
        k=5, metric=DistanceMetric.MANHATTAN, algorithm=SearchAlgorithm.BRUTE_FORCE
    )
    print("Training KNN (Brute-Force / Manhattan)...")
    knn_br.fit(X_train, y_train)
    preds_br = knn_br.predict(X_test)

    accuracy_br = accuracy_score(y_true=y_test, y_pred=preds_br)
    print(f"Brute-Force Accuracy: {accuracy_br * 100:.2f}%\n")

    print("\n" + "-" * 50)
    print("Training DecisionTreeClassifier...")
    dt = DecisionTreeClassifier(max_depth=5)
    dt.fit(X_train, y_train)
    preds_dt = dt.predict(X_test)

    accuracy_dt = accuracy_score(y_true=y_test, y_pred=preds_dt)
    print(f"Decision Tree Accuracy: {accuracy_dt * 100:.2f}%\n")

    print("Sample of 15 Predictions (DT vs Actual):")
    for i in range(min(15, len(y_test))):
        match = "✓" if preds_dt[i] == y_test[i] else "✗"
        print(f"[{match}] Predicted: {preds_dt[i]:>10} | Actual: {y_test[i]:>10}")

    print("\n" + "-" * 50)
    print("Training RandomForestClassifier...")
    rf = RandomForestClassifier(n_estimators=10, max_depth=5)
    rf.fit(X_train, y_train)
    preds_rf = rf.predict(X_test)

    accuracy_rf = accuracy_score(y_true=y_test, y_pred=preds_rf)
    print(f"Random Forest Accuracy: {accuracy_rf * 100:.2f}%\n")

    print("Sample of 15 Predictions (RF vs Actual):")
    for i in range(min(15, len(y_test))):
        match = "✓" if preds_rf[i] == y_test[i] else "✗"
        print(f"[{match}] Predicted: {preds_rf[i]:>10} | Actual: {y_test[i]:>10}")

    print("\n" + "=" * 50)
    print("LINEAR REGRESSION TEST")
    print("=" * 50)

    X_linear = np.array([[-1.0], [-0.5], [0.0], [0.5], [1.0], [1.5]])
    y_linear = np.array([-1.0, 0.5, 2.0, 3.5, 5.0, 6.5])

    lin = LinearRegression(
        learning_rate=0.05,
        max_epochs=5000,
        tol=1e-10,
        schedule=LearningSchedule.CONSTANT,
    )
    lin.fit(X_linear, y_linear)
    lin_preds = lin.predict(X_linear)
    lin_mse = mean_squared_error(y_true=y_linear, y_pred=lin_preds)
    lin_mae = mean_absolute_error(y_true=y_linear, y_pred=lin_preds)
    lin_r2 = r2_score(y_true=y_linear, y_pred=lin_preds)

    print(f"LinearRegression train MSE (constant): {lin_mse:.8f}")
    print(f"LinearRegression train MAE (constant): {lin_mae:.8f}")
    print(f"LinearRegression train R2 (constant): {lin_r2:.8f}")
    print(f"LinearRegression prediction for x=2: {lin.predict(np.array([[2.0]]))[0]:.6f}")

    lin_decay = LinearRegression(
        learning_rate=0.2,
        max_epochs=5000,
        tol=1e-10,
        schedule=LearningSchedule.TIME_DECAY,
    )
    lin_decay.fit(X_linear, y_linear)
    lin_decay_preds = lin_decay.predict(X_linear)
    lin_decay_mse = mean_squared_error(y_true=y_linear, y_pred=lin_decay_preds)
    print(f"LinearRegression train MSE (time decay): {lin_decay_mse:.8f}")

    print("\n" + "=" * 50)
    print("LOGISTIC REGRESSION TEST")
    print("=" * 50)

    X_log = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [0.4, 0.3],
            [1.0, 1.1],
            [1.2, 1.0],
            [1.4, 1.3],
        ]
    )
    y_log = np.array([0, 0, 0, 1, 1, 1])

    log = LogisticRegression(
        learning_rate=0.2,
        max_epochs=5000,
        tol=1e-10,
        schedule=LearningSchedule.EXPONENTIAL,
    )
    log.fit(X_log, y_log)
    log_probs = log.predict_proba(X_log)
    log_preds = log.predict(X_log)
    log_accuracy = accuracy_score(y_true=y_log, y_pred=log_preds)
    log_precision = precision_score(y_true=y_log, y_pred=log_preds)
    log_recall = recall_score(y_true=y_log, y_pred=log_preds)
    log_f1 = f1_score(y_true=y_log, y_pred=log_preds)
    log_cm = confusion_matrix(y_true=y_log, y_pred=log_preds)

    print(f"LogisticRegression train accuracy: {log_accuracy * 100:.2f}%")
    print(f"LogisticRegression train precision (macro): {log_precision:.4f}")
    print(f"LogisticRegression train recall (macro): {log_recall:.4f}")
    print(f"LogisticRegression train F1 (macro): {log_f1:.4f}")
    print(f"LogisticRegression probabilities: {np.round(log_probs, 4)}")
    print(f"LogisticRegression confusion matrix:\n{log_cm}")

    print("\n" + "=" * 50)
    print("METRICS PACKAGE SANITY CHECKS")
    print("=" * 50)

    y_true_cls = np.array([0, 1, 2, 2, 1, 0])
    y_pred_cls = np.array([0, 2, 2, 2, 1, 0])

    cls_accuracy = accuracy_score(y_true=y_true_cls, y_pred=y_pred_cls)
    cls_precision = precision_score(y_true=y_true_cls, y_pred=y_pred_cls)
    cls_recall = recall_score(y_true=y_true_cls, y_pred=y_pred_cls)
    cls_f1 = f1_score(y_true=y_true_cls, y_pred=y_pred_cls)
    cls_cm = confusion_matrix(y_true=y_true_cls, y_pred=y_pred_cls)

    print(f"Classification accuracy: {cls_accuracy:.12f}")
    print(f"Classification precision (macro): {cls_precision:.12f}")
    print(f"Classification recall (macro): {cls_recall:.12f}")
    print(f"Classification F1 (macro): {cls_f1:.12f}")
    print(f"Classification confusion matrix:\n{cls_cm}")

    y_true_reg = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred_reg = np.array([2.5, 0.0, 2.0, 8.0])

    reg_mae = mean_absolute_error(y_true=y_true_reg, y_pred=y_pred_reg)
    reg_mse = mean_squared_error(y_true=y_true_reg, y_pred=y_pred_reg)
    reg_r2 = r2_score(y_true=y_true_reg, y_pred=y_pred_reg)

    print(f"Regression MAE: {reg_mae:.12f}")
    print(f"Regression MSE: {reg_mse:.12f}")
    print(f"Regression R2: {reg_r2:.12f}")

    assert (
        lin_mse < 1e-8
    ), "LinearRegression (constant schedule) did not converge as expected"
    assert lin_mae < 1e-4, "LinearRegression MAE is unexpectedly high"
    assert lin_r2 > 0.999999, "LinearRegression R2 is unexpectedly low"
    assert (
        lin_decay_mse < 2e-2
    ), "LinearRegression (time decay schedule) did not converge as expected"
    assert log_accuracy >= 1.0, "LogisticRegression failed on linearly separable toy data"
    assert log_precision >= 1.0, "LogisticRegression precision should be perfect"
    assert log_recall >= 1.0, "LogisticRegression recall should be perfect"
    assert log_f1 >= 1.0, "LogisticRegression F1 should be perfect"
    assert np.array_equal(
        log_cm, np.array([[3, 0], [0, 3]])
    ), "LogisticRegression confusion matrix mismatch"

    assert np.isclose(cls_accuracy, 5.0 / 6.0), "accuracy_score sanity check failed"
    assert np.isclose(cls_precision, 8.0 / 9.0), "precision_score sanity check failed"
    assert np.isclose(cls_recall, 5.0 / 6.0), "recall_score sanity check failed"
    assert np.isclose(cls_f1, 0.8222222222222223), "f1_score sanity check failed"
    assert np.array_equal(
        cls_cm, np.array([[2, 0, 0], [0, 1, 1], [0, 0, 2]])
    ), "confusion_matrix sanity check failed"

    assert np.isclose(reg_mae, 0.5), "mean_absolute_error sanity check failed"
    assert np.isclose(reg_mse, 0.375), "mean_squared_error sanity check failed"
    assert np.isclose(reg_r2, 0.9486081370449679), "r2_score sanity check failed"


if __name__ == "__main__":
    main()
