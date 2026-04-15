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
from glassbox.models import (
    DecisionTreeClassifier,
    DistanceMetric,
    KNeighborsClassifier,
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

    accuracy = np.mean(preds == y_test)
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

    accuracy_br = np.mean(preds_br == y_test)
    print(f"Brute-Force Accuracy: {accuracy_br * 100:.2f}%\n")

    print("\n" + "-" * 50)
    print("Training DecisionTreeClassifier...")
    dt = DecisionTreeClassifier(max_depth=5)
    dt.fit(X_train, y_train)
    preds_dt = dt.predict(X_test)

    accuracy_dt = np.mean(preds_dt == y_test)
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

    accuracy_rf = np.mean(preds_rf == y_test)
    print(f"Random Forest Accuracy: {accuracy_rf * 100:.2f}%\n")

    print("Sample of 15 Predictions (RF vs Actual):")
    for i in range(min(15, len(y_test))):
        match = "✓" if preds_rf[i] == y_test[i] else "✗"
        print(f"[{match}] Predicted: {preds_rf[i]:>10} | Actual: {y_test[i]:>10}")


if __name__ == "__main__":
    main()
