# Project 2: Data Classification Using AI

**Author:** Saniya Inam
**Internship:** DecodeLabs — AI Track (Batch 2026)
**Track:** Artificial Intelligence Engineer, Industrial Training Kit

## Overview

This project builds a basic **supervised learning** classification model that learns to recognize patterns in labeled data and categorize new, unseen samples. Unlike Project 1's rule-based chatbot (explicit if-else logic), this system doesn't have hardcoded rules — it learns a decision boundary directly from historical data (the "training set") and applies that learned logic to make predictions.

The dataset used is the classic **Iris dataset**: 150 flower samples across 3 species (Setosa, Versicolor, Virginica), described by 4 features (sepal length, sepal width, petal length, petal width).

## How to Run

1. Make sure Python 3 is installed (`python --version` to check).
2. Install the required library:
   ```
   pip install scikit-learn
   ```
3. Open a terminal in the project folder and run:
   ```
   python data_classification_knn.py
   ```
4. The script will print the full pipeline output: K-tuning results, accuracy/F1/confusion matrix, an algorithm comparison, and predictions on brand-new sample data.

## Pipeline (IPO Framework)

| Stage | What happens |
|---|---|
| **Input** | Load the Iris dataset (150 samples, 4 features, 3 classes) |
| **Process** | Scale features → split into train/test sets → train a K-Nearest Neighbors (KNN) classifier |
| **Output** | Evaluate with accuracy, confusion matrix, and F1 score |

## Design Choices

**Feature scaling (StandardScaler)**
KNN classifies points based on distance to their nearest neighbors. If one feature (e.g. petal length in cm) has a much larger numeric range than another, it would unfairly dominate the distance calculation. `StandardScaler` transforms all features to have mean 0 and variance 1, putting them on equal footing. The scaler is fit only on the training data and then applied to the test data, to avoid leaking information from the test set.

**Train/test split with stratification**
Data is shuffled and split 80/20 to remove any ordering bias, and `stratify=y` ensures both the training and test sets keep the same proportion of each species — important with only 150 total samples split across 3 classes.

**K-Nearest Neighbors (KNN)**
KNN works on the "proximity principle": a new data point is classified by majority vote among its K closest neighbors in the training data. It's simple, has no complex training phase (it just memorizes the data), and works well on small, clean datasets like Iris.

**Choosing K**
Very small K (e.g. K=1) can overfit to noise in the training data; very large K can underfit and produce overly generic predictions. The script automatically tests K values from 1–15 and selects the one with the best test accuracy (the "elbow point").

**Evaluating beyond accuracy**
Accuracy alone can be misleading, especially on imbalanced datasets — a model that always predicts the majority class can still score high "accuracy" while being useless. This project also reports:
- **Confusion matrix** — shows exactly which classes get confused with which (true/false positives and negatives per class)
- **F1 score** — the harmonic mean of precision and recall, giving a more balanced view of performance than accuracy alone

## Bonus Features

**Algorithm comparison**
Beyond KNN, the script also trains Logistic Regression, a Decision Tree, and a Support Vector Machine (SVM) on the same data and compares their accuracy and F1 scores side by side, reporting which performed best on that particular train/test split.

**Testing on completely new, unseen data**
Three made-up flower measurements (not part of the original 150 Iris samples) are fed through the trained model to check that it generalizes correctly rather than just memorizing the training set. All three were classified into their expected species.

## Project Structure

```
data_classification_knn.py   # main script
README.md                     # this file
```

## Key Concepts Demonstrated

- Supervised learning fundamentals (train/test split, fitting, predicting)
- Feature scaling and why it matters for distance-based algorithms
- The K-Nearest Neighbors algorithm and hyperparameter tuning (choosing K)
- Model evaluation beyond raw accuracy (confusion matrix, F1 score)
- Comparing multiple algorithms on the same task
- Testing model generalization on unseen data

## Possible Future Extensions

- Visualize the decision boundaries or confusion matrix as a heatmap
- Try cross-validation instead of a single train/test split for more robust K selection
- Move from tabular data (Iris) toward image-based classification (computer vision) as a next step in the AI track
