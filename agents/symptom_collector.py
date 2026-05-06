from knowledge.disease_rules import SYMPTOMS

SYMPTOM_SYNONYMS = {
    "high temperature": "fever",
    "temperature": "fever",
    "feel hot": "fever",
    "tired": "fatigue",
    "exhausted": "fatigue",
    "runny nose": "runny_nose",
    "stuffy nose": "runny_nose",
    "sore throat": "sore_throat",
    "throat pain": "sore_throat",
    "chest pain": "chest_pain",
    "chest discomfort": "chest_pain",
    "short of breath": "shortness_of_breath",
    "shortness of breath": "shortness_of_breath",
    "breathless": "shortness_of_breath",
    "sensitivity to light": "light_sensitivity",
    "light sensitive": "light_sensitivity",
    "body ache": "body_aches",
    "muscle pain": "body_aches",
    "frequent urination": "frequent_urination",
    "peeing a lot": "frequent_urination",
    "excessive thirst": "excessive_thirst",
    "very thirsty": "excessive_thirst",
    "weight gain": "weight_gain",
    "cold intolerance": "cold_intolerance",
    "always cold": "cold_intolerance",
    "dry skin": "dry_skin",
    "memory issues": "memory_issues",
    "forgetful": "memory_issues",
    "muscle weakness": "muscle_weakness",
    "weak muscles": "muscle_weakness",
    "jaw pain": "jaw_pain",
    "arm pain": "arm_pain",
    "neck stiffness": "neck_stiffness",
    "stiff neck": "neck_stiffness",
    "blurred vision": "blurred_vision",
    "blurry vision": "blurred_vision",
    "heart pounding": "palpitations",
    "heart racing": "palpitations",
    "loss of appetite": "loss_of_appetite",
    "not hungry": "loss_of_appetite",
}


class SymptomCollectionAgent:
    MIN_SYMPTOMS = 2

    def extract_symptoms(self, text: str) -> list:
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
        return len(symptoms) >= self.MIN_SYMPTOMS

    def get_followup_question(self, symptoms: list) -> str:
        if not symptoms:
            return "Please describe your symptoms. What are you experiencing?"
        listed = ", ".join(symptoms)
        return f"I've noted: {listed}. Are there any other symptoms you're experiencing?"
