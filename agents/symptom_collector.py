"""
SymptomCollectionAgent — NLP Symptom Extractor and Follow-up Generator
=======================================================================

AI Technique: Rule-based Natural Language Processing with synonym mapping

Role in pipeline:
    The SymptomCollectionAgent is the first specialist invoked by the
    OrchestratorAgent.  Its responsibility is to parse free-text patient
    descriptions and convert them into a canonical set of normalised symptom
    tokens that downstream agents (DiagnosisAgent in particular) can reason
    over.

    The agent operates in two phases:

    1. **Synonym resolution** — The patient's raw text is scanned for known
       colloquial phrases (e.g. "heart racing", "feel sick", "stiff neck")
       and mapped to a controlled medical vocabulary defined in
       ``SYMPTOM_SYNONYMS``.  Phrases are matched longest-first to avoid
       shorter tokens incorrectly shadowing multi-word expressions.

    2. **Direct token matching** — Any remaining text is compared against the
       canonical symptom list imported from
       ``knowledge.disease_rules.SYMPTOMS`` (both underscore and
       space-separated forms), capturing symptoms the patient stated in
       standard terminology.

    If fewer than ``MIN_SYMPTOMS`` unique symptoms have been collected across
    all turns, the agent generates a contextual follow-up question to elicit
    additional information before the pipeline may proceed to diagnosis.
"""

from knowledge.disease_rules import SYMPTOMS

SYMPTOM_SYNONYMS = {
    # Fever
    "high temperature": "fever",
    "temperature": "fever",
    "feel hot": "fever",
    "feeling hot": "fever",
    "running a fever": "fever",
    "feverish": "fever",
    # Fatigue
    "tired": "fatigue",
    "tiredness": "fatigue",
    "exhausted": "fatigue",
    "exhaustion": "fatigue",
    "weakness": "fatigue",
    "weak": "fatigue",
    "lethargic": "fatigue",
    "no energy": "fatigue",
    "low energy": "fatigue",
    "stressed": "fatigue",
    "stressful": "fatigue",
    "stress": "fatigue",
    "heavy eyes": "fatigue",
    "eyes heavy": "fatigue",
    "eyes are heavy": "fatigue",
    "not feeling well": "fatigue",
    "unwell": "fatigue",
    # Headache
    "head ache": "headache",
    "headche": "headache",
    "headach": "headache",
    "head pain": "headache",
    "head is paining": "headache",
    "head hurts": "headache",
    "head is hurting": "headache",
    "pain in head": "headache",
    "migraine": "headache",
    # Nausea
    "feel like vomiting": "nausea",
    "want to vomit": "nausea",
    "feel sick": "nausea",
    "feeling sick": "nausea",
    "upset stomach": "nausea",
    "queasy": "nausea",
    # Runny nose
    "runny nose": "runny_nose",
    "stuffy nose": "runny_nose",
    "blocked nose": "runny_nose",
    "nose running": "runny_nose",
    # Sore throat
    "sore throat": "sore_throat",
    "throat pain": "sore_throat",
    "throat hurts": "sore_throat",
    "throat is sore": "sore_throat",
    "painful throat": "sore_throat",
    # Chest
    "chest pain": "chest_pain",
    "chest discomfort": "chest_pain",
    "chest tightness": "chest_pain",
    "chest is tight": "chest_pain",
    "pain in chest": "chest_pain",
    # Breathing
    "short of breath": "shortness_of_breath",
    "shortness of breath": "shortness_of_breath",
    "breathless": "shortness_of_breath",
    "difficulty breathing": "shortness_of_breath",
    "hard to breathe": "shortness_of_breath",
    "can't breathe": "shortness_of_breath",
    # Light sensitivity
    "sensitivity to light": "light_sensitivity",
    "light sensitive": "light_sensitivity",
    "sensitive to light": "light_sensitivity",
    "light bothers": "light_sensitivity",
    # Body aches
    "body ache": "body_aches",
    "body aches": "body_aches",
    "muscle pain": "body_aches",
    "aching body": "body_aches",
    "whole body hurts": "body_aches",
    "body is aching": "body_aches",
    # Dizziness
    "dizzy": "dizziness",
    "feel dizzy": "dizziness",
    "lightheaded": "dizziness",
    "light headed": "dizziness",
    "spinning": "dizziness",
    "vertigo": "dizziness",
    # Urination
    "frequent urination": "frequent_urination",
    "peeing a lot": "frequent_urination",
    "urinating a lot": "frequent_urination",
    "going to bathroom a lot": "frequent_urination",
    # Thirst
    "excessive thirst": "excessive_thirst",
    "very thirsty": "excessive_thirst",
    "always thirsty": "excessive_thirst",
    "thirsty all the time": "excessive_thirst",
    # Other
    "weight gain": "weight_gain",
    "gaining weight": "weight_gain",
    "cold intolerance": "cold_intolerance",
    "always cold": "cold_intolerance",
    "feel cold always": "cold_intolerance",
    "dry skin": "dry_skin",
    "memory issues": "memory_issues",
    "forgetful": "memory_issues",
    "forgetting things": "memory_issues",
    "muscle weakness": "muscle_weakness",
    "weak muscles": "muscle_weakness",
    "jaw pain": "jaw_pain",
    "arm pain": "arm_pain",
    "neck stiffness": "neck_stiffness",
    "stiff neck": "neck_stiffness",
    "blurred vision": "blurred_vision",
    "blurry vision": "blurred_vision",
    "vision is blurry": "blurred_vision",
    "can't see clearly": "blurred_vision",
    "heart pounding": "palpitations",
    "heart racing": "palpitations",
    "fast heartbeat": "palpitations",
    "heart beating fast": "palpitations",
    "loss of appetite": "loss_of_appetite",
    "not hungry": "loss_of_appetite",
    "no appetite": "loss_of_appetite",
    "don't feel like eating": "loss_of_appetite",
    # Sweating / chills
    "sweating": "sweating",
    "sweat a lot": "sweating",
    "night sweats": "sweating",
    "shivering": "chills",
    "chilly": "chills",
    "feeling cold": "chills",
    # Cough
    "coughing": "cough",
    "dry cough": "cough",
    "wet cough": "cough",
    "bad cough": "cough",
    # Vomiting
    "vomiting": "vomiting",
    "throwing up": "vomiting",
    "puking": "vomiting",
}


class SymptomCollectionAgent:
    """NLP agent that extracts and normalises symptoms from patient free text.

    This agent bridges the gap between colloquial patient language and the
    controlled medical vocabulary used by the knowledge base.  It maintains
    no per-instance state; all session accumulation is performed by the
    OrchestratorAgent, which passes the growing symptom list back into this
    agent for follow-up question generation.

    Class attributes
    ----------------
    MIN_SYMPTOMS : int
        Minimum number of distinct canonical symptoms required before the
        pipeline is permitted to advance to the diagnosis stage.
    """

    MIN_SYMPTOMS = 2

    def extract_symptoms(self, text: str) -> list:
        """Parse free-text input and return a sorted list of canonical symptom tokens.

        The extraction proceeds in two ordered passes:

        1. **Synonym pass** — Iterates over ``SYMPTOM_SYNONYMS`` entries sorted
           by descending phrase length so that longer, more specific phrases
           (e.g. ``"shortness of breath"``) are matched before shorter
           sub-phrases (e.g. ``"breath"``).  Each matched phrase is consumed
           from the working copy of the text to prevent double-counting.

        2. **Direct token pass** — Scans the remaining text for canonical
           symptom identifiers (both underscore form ``"runny_nose"`` and
           space form ``"runny nose"``) drawn from the knowledge-base symptom
           registry.

        Parameters
        ----------
        text : str
            Raw patient input string (any case).

        Returns
        -------
        list[str]
            Alphabetically sorted list of unique canonical symptom strings
            (e.g. ``["fever", "headache", "nausea"]``).  Returns an empty
            list when ``text`` is falsy.
        """
        if not text:
            return []
        text_lower = text.lower()
        found = set()

        for phrase, canonical in sorted(SYMPTOM_SYNONYMS.items(), key=lambda x: -len(x[0])):
            if phrase in text_lower:
                found.add(canonical)
                text_lower = text_lower.replace(phrase, "")

        for symptom in SYMPTOMS:
            normalized = symptom.replace("_", " ")
            if normalized in text_lower or symptom in text_lower:
                found.add(symptom)

        return sorted(found)

    def has_enough_symptoms(self, symptoms: list) -> bool:
        """Return True when the symptom list meets the minimum threshold for diagnosis.

        Parameters
        ----------
        symptoms : list[str]
            Accumulated canonical symptom list from the current session.

        Returns
        -------
        bool
            ``True`` if ``len(symptoms) >= MIN_SYMPTOMS``, ``False`` otherwise.
        """
        return len(symptoms) >= self.MIN_SYMPTOMS

    def get_followup_question(self, symptoms: list) -> str:
        """Generate a context-sensitive follow-up question to elicit more symptom detail.

        When no symptoms have been collected yet, returns a generic opening
        prompt.  When at least one symptom is known, lists them and asks
        whether the patient has any additional symptoms, giving the user
        visibility into what has already been captured.

        Parameters
        ----------
        symptoms : list[str]
            Canonical symptoms collected so far in the current session.

        Returns
        -------
        str
            A natural-language follow-up question suitable for direct display
            in the chat interface.
        """
        if not symptoms:
            return "Please describe your symptoms. What are you experiencing?"
        listed = ", ".join(symptoms)
        return f"I've noted: {listed}. Are there any other symptoms you're experiencing?"
