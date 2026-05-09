"""
DiagnosisAgent — Hybrid FOL + Naive Bayes Diagnostic Reasoner
==============================================================

AI Techniques:
    * First-Order Logic (FOL) inference over a hand-crafted disease rule base
    * Naive Bayes probabilistic classification (scikit-learn, loaded from a
      pre-trained ``model.pkl`` artefact)
    * Weighted linear blending of symbolic and statistical confidence scores

Role in pipeline:
    The DiagnosisAgent is invoked by the OrchestratorAgent after the
    SymptomCollectionAgent has gathered a sufficient symptom set.  It produces
    a ranked list of up to three candidate diagnoses, each annotated with:

    * The raw FOL score (rule-coverage ratio).
    * The Naive Bayes posterior probability for the disease class.
    * A blended confidence score (FOL_WEIGHT × fol_score + ML_WEIGHT × ml_prob)
      used to rank candidates and gate pipeline progression.

    The hybrid approach deliberately prioritises the interpretable FOL layer
    (weight 0.6) while using the statistical model as a calibration signal
    (weight 0.4).  This means the system will only promote a diagnosis that
    satisfies at least one *required* symptom rule — the ML layer alone cannot
    surface a disease that has no FOL evidence.

    If no pre-trained model is found at ``MODEL_PATH``, the agent falls back
    gracefully to pure FOL inference (ML weight effectively becomes zero for
    all candidates).
"""

import os
import numpy as np
from knowledge.disease_rules import DISEASE_RULES, SYMPTOMS

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "model.pkl")


class DiagnosisAgent:
    """Hybrid diagnostic agent combining FOL rule inference with Naive Bayes ML.

    The agent maintains a single optional reference to a pre-trained
    scikit-learn Naive Bayes classifier.  All other knowledge is encoded in
    the ``DISEASE_RULES`` dictionary imported from the knowledge base, which
    maps each disease to its required and supporting symptom sets.

    Class attributes
    ----------------
    ML_WEIGHT : float
        Fraction of the blended confidence score contributed by the Naive
        Bayes posterior probability (default 0.4).
    FOL_WEIGHT : float
        Fraction of the blended confidence score contributed by the FOL
        rule-coverage ratio (default 0.6).

    Note: ``ML_WEIGHT + FOL_WEIGHT`` must equal 1.0 for the blended score
    to remain in the [0, 1] range.
    """

    ML_WEIGHT = 0.4
    FOL_WEIGHT = 0.6

    def __init__(self):
        """Initialise the agent and attempt to load the pre-trained ML model."""
        self._model = None
        self._load_model()

    def _load_model(self):
        """Load the serialised Naive Bayes model from disk if available.

        Uses ``joblib`` for deserialisation.  If the model file does not exist
        at ``MODEL_PATH``, ``self._model`` remains ``None`` and the agent
        operates in FOL-only mode without raising an error.
        """
        if os.path.exists(MODEL_PATH):
            import joblib
            self._model = joblib.load(MODEL_PATH)

    def run_fol_inference(self, symptoms: list) -> list:
        """Score candidate diseases against the patient's symptoms using FOL rules.

        For each disease in ``DISEASE_RULES``, the method computes a coverage
        score that reflects how well the patient's symptom set satisfies the
        disease's logical preconditions:

        * A disease is *eligible* only if at least one of its required symptoms
          is present (hard gate).
        * The FOL score is a weighted sum:

          .. code-block:: text

              score = (required_matches / |required|) × 0.7
                    + (supporting_matches / |supporting|) × 0.3

          The 70 / 30 split encodes the domain assumption that required
          symptoms are diagnostically more informative than supporting ones.

        Parameters
        ----------
        symptoms : list[str]
            Canonical symptom tokens collected from the patient.

        Returns
        -------
        list[dict]
            Up to three highest-scoring disease dicts, each containing:
            ``disease``, ``fol_score``, ``matched_required``,
            ``matched_supporting``.  Sorted descending by ``fol_score``.
        """
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
        """Produce per-disease posterior probabilities using the Naive Bayes model.

        Encodes the patient's symptoms as a binary presence/absence feature
        vector aligned to the global ``SYMPTOMS`` vocabulary, then queries the
        classifier's ``predict_proba`` method.

        Parameters
        ----------
        symptoms : list[str]
            Canonical symptom tokens collected from the patient.

        Returns
        -------
        dict[str, float]
            Mapping of disease class label to posterior probability.  Returns
            an empty dict when no model is loaded (FOL-only fallback mode).
        """
        if self._model is None:
            return {}
        vector = np.array([[1 if s in symptoms else 0 for s in SYMPTOMS]])
        probs = self._model.predict_proba(vector)[0]
        classes = self._model.classes_
        return {cls: float(prob) for cls, prob in zip(classes, probs)}

    def diagnose(self, symptoms: list) -> list:
        """Produce a ranked list of diagnoses by blending FOL and ML scores.

        Orchestrates the two inference passes and merges their outputs:

        1. Calls ``run_fol_inference`` to obtain the FOL-shortlisted candidates.
        2. Calls ``run_ml_inference`` to obtain ML posterior probabilities.
        3. For each FOL candidate, looks up its ML posterior (defaulting to 0.0
           if the disease is not in the ML output) and computes the blended
           confidence: ``FOL_WEIGHT × fol_score + ML_WEIGHT × ml_prob``.
        4. Re-sorts the candidates by blended confidence descending.

        The blended confidence is the primary signal used by the
        OrchestratorAgent to decide whether to proceed to treatment planning or
        request additional symptoms from the patient.

        Parameters
        ----------
        symptoms : list[str]
            Canonical symptom tokens collected from the patient.

        Returns
        -------
        list[dict]
            Candidate diagnosis dicts sorted descending by ``confidence``.
            Each dict contains: ``disease``, ``fol_score``, ``ml_prob``,
            ``confidence``, ``matched_required``, ``matched_supporting``.
        """
        fol_results = self.run_fol_inference(symptoms)
        ml_probs = self.run_ml_inference(symptoms)

        for result in fol_results:
            ml_prob = ml_probs.get(result["disease"], 0.0)
            blended = self.FOL_WEIGHT * result["fol_score"] + self.ML_WEIGHT * ml_prob
            result["ml_prob"] = round(ml_prob, 3)
            result["confidence"] = round(blended, 3)

        return sorted(fol_results, key=lambda x: x["confidence"], reverse=True)
