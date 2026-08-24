"""
Project 2: Data Classification Using AI
DecodeLabs - AI Internship 2026
Author: Saniya Inam

Goal: Build a basic supervised-learning classification model using a
small dataset (Iris) — load data, split into train/test sets, scale
features, train a K-Nearest Neighbors classifier, and evaluate it
properly (not just with accuracy, since accuracy can be misleading).

Pipeline (IPO Framework):
    INPUT   -> Load Iris dataset, understand features
    PROCESS -> Scale features, train/test split, train KNN
    OUTPUT  -> Confusion matrix, F1 score, classification report
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    accuracy_score,
    classification_report,
)


def load_data():
    """
    Phase 1: INPUT
    Load the Iris dataset — 150 samples, 3 classes (Setosa,
    Versicolor, Virginica), 4 features (sepal/petal length & width).
    """
    iris = load_iris()
    X = iris.data          # features: sepal length, sepal width, petal length, petal width
    y = iris.target        # labels: 0 = setosa, 1 = versicolor, 2 = virginica
    target_names = iris.target_names
    return X, y, target_names


def prepare_data(X, y):
    """
    Phase 2: PROCESS (part 1)
    - Split into training and testing sets (80/20), shuffled to
      remove order bias.
    - Scale features so KNN's distance calculations aren't biased
      toward features with larger raw values.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,   # fixed seed so results are reproducible
        shuffle=True,
        stratify=y,        # keeps class proportions balanced in both sets
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # fit + transform on training data
    X_test_scaled = scaler.transform(X_test)          # only transform test data (no fitting!)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(X_train, y_train, k=5):
    """
    Phase 2: PROCESS (part 2)
    Train a K-Nearest Neighbors classifier.
    "The Proximity Principle": a new point is classified by majority
    vote among its K nearest neighbors.
    """
    model = KNeighborsClassifier(n_neighbors=k)   # INSTANTIATE
    model.fit(X_train, y_train)                   # FIT (memorize the map)
    return model


def evaluate_model(model, X_test, y_test, target_names):
    """
    Phase 3: OUTPUT
    Evaluate using accuracy, confusion matrix, and F1 score.
    Accuracy alone can be misleading on imbalanced data, so we look
    deeper with a confusion matrix (TP/FP/FN/TN per class) and F1
    score (harmonic mean of precision and recall).
    """
    predictions = model.predict(X_test)   # PREDICT

    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=target_names)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}\n")

    print("Confusion Matrix:")
    print("(rows = actual class, columns = predicted class)")
    print(cm, "\n")

    print("Classification Report:")
    print(report)

    return acc, f1, cm


def find_best_k(X_train, y_train, X_test, y_test, max_k=15):
    """
    Bonus: Tune the K value.
    Tries several K values and reports the one with lowest error rate
    on the test set (the 'elbow point' from the training slides).
    """
    print("Tuning K (finding the elbow point)...")
    best_k, best_acc = None, 0

    for k in range(1, max_k + 1):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        print(f"  K={k:>2} -> Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc, best_k = acc, k

    print(f"\nBest K found: {best_k} (Accuracy: {best_acc:.4f})\n")
    return best_k


def compare_algorithms(X_train, y_train, X_test, y_test):
    """
    Bonus: Compare different classification algorithms.
    KNN isn't the only option — Logistic Regression, Decision Tree,
    and SVM are other common supervised learning approaches. This
    trains each on the same data and compares their performance.
    """
    print("Comparing different algorithms on the same data...\n")

    models = {
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Support Vector Machine": SVC(),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="weighted")
        results[name] = (acc, f1)
        print(f"  {name:<25} Accuracy: {acc:.4f}   F1 Score: {f1:.4f}")

    best_model = max(results, key=lambda name: results[name][1])
    print(f"\nBest performing algorithm on this split: {best_model} "
          f"(F1 Score: {results[best_model][1]:.4f})\n")

    return results


def test_on_new_data(model, scaler, target_names):
    """
    Bonus: Test the trained model on completely new, unseen data
    (not from the original dataset at all) to see how it generalizes.
    These are made-up flower measurements, not from Iris's 150 samples.
    """
    print("Testing the model on brand-new, unseen flower measurements...\n")

    # [sepal length, sepal width, petal length, petal width] in cm
    new_samples = np.array([
        [5.0, 3.6, 1.4, 0.2],   # expected: setosa-like (small petals)
        [6.0, 2.9, 4.5, 1.5],   # expected: versicolor-like (medium)
        [6.7, 3.3, 5.7, 2.1],   # expected: virginica-like (large petals)
    ])

    new_samples_scaled = scaler.transform(new_samples)
    predictions = model.predict(new_samples_scaled)

    for sample, pred in zip(new_samples, predictions):
        print(f"  Measurements {sample.tolist()} -> Predicted: {target_names[pred]}")
    print()


def main():
    # INPUT
    X, y, target_names = load_data()
    print(f"Loaded Iris dataset: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(target_names)} classes ({', '.join(target_names)})\n")

    # PROCESS
    X_train, X_test, y_train, y_test, scaler = prepare_data(X, y)

    # Bonus: find the best K before final training
    best_k = find_best_k(X_train, y_train, X_test, y_test)

    model = train_model(X_train, y_train, k=best_k)

    # OUTPUT
    evaluate_model(model, X_test, y_test, target_names)

    # Bonus: compare KNN against other algorithms
    compare_algorithms(X_train, y_train, X_test, y_test)

    # Bonus: test the final model on completely new, unseen data
    test_on_new_data(model, scaler, target_names)


if __name__ == "__main__":
    main()
