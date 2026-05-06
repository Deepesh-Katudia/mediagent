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
