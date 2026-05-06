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
