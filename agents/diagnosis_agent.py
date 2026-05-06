import os
import numpy as np
from knowledge.disease_rules import DISEASE_RULES, SYMPTOMS

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "model.pkl")


class DiagnosisAgent:
    ML_WEIGHT = 0.4
    FOL_WEIGHT = 0.6

    def __init__(self):
        self._model = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(MODEL_PATH):
            import joblib
            self._model = joblib.load(MODEL_PATH)

    def run_fol_inference(self, symptoms: list) -> list:
        symptom_set = set(symptoms)
        results = []

        for disease, rules in DISEASE_RULES.items():
            required = set(rules["required"])
            supporting = set(rules["supporting"])

            required_matches = len(required & symptom_set)
            if required_matches == 0:
                continue

            supporting_matches = len(supporting & symptom_set)
            score = (
                (required_matches / len(required)) * 0.7
                + (supporting_matches / max(len(supporting), 1)) * 0.3
            )

            results.append({
                "disease": disease,
                "fol_score": round(score, 3),
                "matched_required": sorted(required & symptom_set),
                "matched_supporting": sorted(supporting & symptom_set),
            })

        return sorted(results, key=lambda x: x["fol_score"], reverse=True)[:3]

    def run_ml_inference(self, symptoms: list) -> dict:
        if self._model is None:
            return {}
        vector = np.array([[1 if s in symptoms else 0 for s in SYMPTOMS]])
        probs = self._model.predict_proba(vector)[0]
        classes = self._model.classes_
        return {cls: float(prob) for cls, prob in zip(classes, probs)}

    def diagnose(self, symptoms: list) -> list:
        fol_results = self.run_fol_inference(symptoms)
        ml_probs = self.run_ml_inference(symptoms)

        for result in fol_results:
            ml_prob = ml_probs.get(result["disease"], 0.0)
            blended = self.FOL_WEIGHT * result["fol_score"] + self.ML_WEIGHT * ml_prob
            result["ml_prob"] = round(ml_prob, 3)
            result["confidence"] = round(blended, 3)

        return sorted(fol_results, key=lambda x: x["confidence"], reverse=True)
