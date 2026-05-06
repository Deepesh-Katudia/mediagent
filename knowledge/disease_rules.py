SYMPTOMS = [
    "fever", "cough", "fatigue", "body_aches", "runny_nose", "sore_throat",
    "chest_pain", "shortness_of_breath", "headache", "nausea", "vomiting",
    "light_sensitivity", "dizziness", "palpitations", "sweating", "chills",
    "loss_of_appetite", "frequent_urination", "excessive_thirst", "blurred_vision",
    "weight_gain", "cold_intolerance", "constipation", "dry_skin", "neck_stiffness",
    "jaw_pain", "arm_pain", "indigestion", "muscle_weakness", "memory_issues"
]

DISEASE_RULES = {
    "Flu": {
        "required": ["fever", "body_aches", "fatigue"],
        "supporting": ["chills", "headache", "cough", "sore_throat"]
    },
    "Common Cold": {
        "required": ["runny_nose", "sore_throat"],
        "supporting": ["cough", "fatigue", "headache"]
    },
    "Pneumonia": {
        "required": ["fever", "cough", "chest_pain"],
        "supporting": ["shortness_of_breath", "fatigue", "sweating", "chills"]
    },
    "Bronchitis": {
        "required": ["cough", "fatigue"],
        "supporting": ["chest_pain", "shortness_of_breath", "fever"]
    },
    "Hypertension": {
        "required": ["headache", "dizziness"],
        "supporting": ["palpitations", "shortness_of_breath", "chest_pain"]
    },
    "Angina": {
        "required": ["chest_pain", "shortness_of_breath"],
        "supporting": ["jaw_pain", "arm_pain", "sweating", "nausea", "dizziness"]
    },
    "Type 2 Diabetes": {
        "required": ["frequent_urination", "excessive_thirst"],
        "supporting": ["blurred_vision", "fatigue", "weight_gain", "loss_of_appetite"]
    },
    "Hypothyroidism": {
        "required": ["fatigue", "weight_gain", "cold_intolerance"],
        "supporting": ["constipation", "dry_skin", "memory_issues", "muscle_weakness"]
    },
    "Migraine": {
        "required": ["headache", "nausea", "light_sensitivity"],
        "supporting": ["vomiting", "dizziness", "fatigue"]
    },
    "Tension Headache": {
        "required": ["headache"],
        "supporting": ["neck_stiffness", "fatigue", "dizziness"]
    }
}
