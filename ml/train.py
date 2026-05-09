"""
train.py
========
Training script for the Naive Bayes disease classifier used by DiagnosisAgent
(Russell & Norvig, Ch. 20).

This script:
1. Calls dataset_generator.generate_dataset() to produce a synthetic labelled
   symptom dataset derived from the FOL disease rules.
2. Splits the data into training (80%) and test (20%) partitions.
3. Fits a MultinomialNB model on the binary symptom feature vectors.
4. Asserts that test accuracy meets the 0.75 minimum threshold before saving.
5. Serialises the trained model to ml/model.pkl via joblib so that
   DiagnosisAgent can load it at runtime without re-training on each request.

Run directly with:
    python ml/train.py

AI technique supported: Naive Bayes ML Classifier (DiagnosisAgent).
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from knowledge.disease_rules import SYMPTOMS
from ml.dataset_generator import generate_dataset

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")


def train():
    """Train the Naive Bayes classifier and persist it to disk.

    Generates a synthetic dataset, splits it into train/test sets, trains a
    MultinomialNB model, validates accuracy against a 0.75 threshold, and
    saves the serialised model to MODEL_PATH (ml/model.pkl).

    Raises:
        AssertionError: If the test set accuracy falls below 0.75.
    """
    df = generate_dataset(samples_per_disease=1000)
    X = df[SYMPTOMS].values
    y = df["disease"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Test accuracy: {accuracy:.3f}")
    assert accuracy >= 0.75, f"Model accuracy {accuracy:.3f} below 0.75 threshold"

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
