"""
dataset_generator.py
====================
Synthetic training data generator for the Naive Bayes disease classifier
(Russell & Norvig, Ch. 20).

This module programmatically constructs a labelled symptom dataset from the
FOL disease rules in knowledge/disease_rules.py, removing the need for a
real patient database during development and academic demonstration.

Each generated sample:
- Sets all required symptoms for the target disease to 1.
- Randomly activates each supporting symptom with 60% probability, reflecting
  real-world clinical variability.
- Applies random bit-flipping noise at a configurable rate (default 10%) to
  prevent the model from memorising a perfectly clean rule set.

AI technique supported: Naive Bayes ML Classifier (DiagnosisAgent).
"""

import numpy as np
import pandas as pd
from knowledge.disease_rules import DISEASE_RULES, SYMPTOMS


def generate_dataset(samples_per_disease: int = 1000, noise_prob: float = 0.1, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic symptom-disease dataset for classifier training.

    For each disease defined in DISEASE_RULES, produces `samples_per_disease`
    labelled rows. Each row is a binary feature vector over SYMPTOMS, where 1
    indicates the symptom is present and 0 indicates it is absent.

    Args:
        samples_per_disease: Number of training rows to create per disease
            class. Defaults to 1000.
        noise_prob: Probability of randomly flipping any individual symptom
            bit to introduce realistic noise. Defaults to 0.1 (10%).
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        pd.DataFrame: A DataFrame with one column per symptom (binary int)
        plus a 'disease' string column containing the class label.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for disease, rules in DISEASE_RULES.items():
        for _ in range(samples_per_disease):
            row = {s: 0 for s in SYMPTOMS}

            for s in rules["required"]:
                row[s] = 1

            for s in rules["supporting"]:
                if rng.random() > 0.4:
                    row[s] = 1

            for s in SYMPTOMS:
                if rng.random() < noise_prob:
                    row[s] = 1 - row[s]

            row["disease"] = disease
            rows.append(row)

    return pd.DataFrame(rows)
