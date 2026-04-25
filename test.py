import json

import numpy as np

from glassbox.models import (
    DecisionTreeClassifier,
    DistanceMetric,
    GaussianNB,
    KNeighborsClassifier,
    LearningSchedule,
    LinearRegression,
    LogisticRegression,
    RandomForestClassifier,
    SearchAlgorithm,
)


def main():
    # Create synthetic test data instead of loading penguins
    np.random.seed(42)
    
    # Create a simple 3-class dataset
    n_per_class = 50
    n_features = 6
    
    # Class 0: centered around origin
    X_class0 = np.random.randn(n_per_class, n_features) * 0.5
    y_class0 = np.array([0] * n_per_class)
    
    # Class 1: centered at (2, 2, ...)
    X_class1 = np.random.randn(n_per_class, n_features) * 0.5 + 2.0
    y_class1 = np.array([1] * n_per_class)
    
    # Class 2: centered at (-2, -2, ...)
    X_class2 = np.random.randn(n_per_class, n_features) * 0.5 - 2.0
    y_class2 = np.array([2] * n_per_class)
    
    X = np.vstack([X_class0, X_class1, X_class2])
    y = np.hstack([y_class0, y_class1, y_class2])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Train/test split (80/20)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training on {len(X_train)} samples. Testing on {len(X_test)} samples.\n")
    print("=" * 50)
    print("GAUSSIAN NB CLASSIFIER TEST")
    print("=" * 50)
    
    gnb = GaussianNB(epsilon=1e-9)
    print("Training GaussianNB...")
    gnb.fit(X_train, y_train)
    
    preds_gnb = gnb.predict(X_test)
    proba_gnb = gnb.predict_proba(X_test)
    
    accuracy_gnb = np.mean(preds_gnb == y_test)
    print(f"GaussianNB Accuracy: {accuracy_gnb * 100:.2f}%\n")
    
    print("Sample of 15 Predictions (GaussianNB vs Actual):")
    for i in range(min(15, len(y_test))):
        match = "✓" if preds_gnb[i] == y_test[i] else "✗"
        prob_max = np.max(proba_gnb[i])
        print(f"[{match}] Predicted: {int(preds_gnb[i])} | Actual: {int(y_test[i])} | Confidence: {prob_max:.4f}")
    
    # Verify probabilities sum to 1
    prob_sums = np.sum(proba_gnb, axis=1)
    assert np.allclose(prob_sums, 1.0), "Probabilities do not sum to 1"
    print("\n✓ Probabilities sum to 1.0")
    
    # Verify predictions match highest probability class
    pred_from_proba = np.argmax(proba_gnb, axis=1)
    assert np.allclose(preds_gnb, pred_from_proba), "Predictions don't match highest probability"
    print("✓ Predictions match highest probability class")
    
    # Verify model was fitted
    assert len(gnb.classes) == 3, "Classes not detected correctly"
    print(f"✓ Detected {len(gnb.classes)} classes: {gnb.classes}")
    
    print("\n" + "=" * 50)
    print("GAUSSIAN NB TEST: EDGE CASES")
    print("=" * 50)
    
    # Test error on unfitted model
    unfitted_gnb = GaussianNB()
    try:
        unfitted_gnb.predict(X_test)
        print("✗ Should have raised ValueError for unfitted model")
    except ValueError as e:
        print(f"✓ Correctly raised error for unfitted model: {e}")
    
    # Test error on mismatched dimensions
    try:
        gnb.fit(X_train, y_train[:5])
        print("✗ Should have raised ValueError for mismatched dimensions")
    except ValueError as e:
        print(f"✓ Correctly raised error for mismatched dimensions: {e}")
    
    print("\n" + "=" * 50)
    print("KNN CLASSIFIER TEST (Train/Test Split)")
    print("=" * 50)

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
    lin_mse = np.mean((lin_preds - y_linear) ** 2)

    print(f"LinearRegression train MSE (constant): {lin_mse:.8f}")
    print(f"LinearRegression prediction for x=2: {lin.predict(np.array([[2.0]]))[0]:.6f}")

    lin_decay = LinearRegression(
        learning_rate=0.2,
        max_epochs=5000,
        tol=1e-10,
        schedule=LearningSchedule.TIME_DECAY,
    )
    lin_decay.fit(X_linear, y_linear)
    lin_decay_mse = np.mean((lin_decay.predict(X_linear) - y_linear) ** 2)
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
    log_accuracy = np.mean(log_preds == y_log)

    print(f"LogisticRegression train accuracy: {log_accuracy * 100:.2f}%")
    print(f"LogisticRegression probabilities: {np.round(log_probs, 4)}")

    assert (
        lin_mse < 1e-8
    ), "LinearRegression (constant schedule) did not converge as expected"
    assert (
        lin_decay_mse < 2e-2
    ), "LinearRegression (time decay schedule) did not converge as expected"
    assert log_accuracy >= 1.0, "LogisticRegression failed on linearly separable toy data"


if __name__ == "__main__":
    main()
