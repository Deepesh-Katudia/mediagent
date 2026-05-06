import numpy as np
import pandas as pd
from knowledge.disease_rules import DISEASE_RULES, SYMPTOMS


def generate_dataset(samples_per_disease: int = 1000, noise_prob: float = 0.1, seed: int = 42) -> pd.DataFrame:
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
