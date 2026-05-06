DISEASE_DRUGS = {
    "Flu": ["Oseltamivir", "Ibuprofen", "Acetaminophen"],
    "Common Cold": ["Decongestants", "Antihistamines", "Acetaminophen"],
    "Pneumonia": ["Amoxicillin", "Azithromycin", "Ibuprofen"],
    "Bronchitis": ["Albuterol", "Ibuprofen", "Guaifenesin"],
    "Hypertension": ["ACE_Inhibitors", "Beta_Blockers", "Amlodipine"],
    "Angina": ["Aspirin", "Beta_Blockers", "Nitroglycerin"],
    "Type 2 Diabetes": ["Metformin", "Insulin", "Sitagliptin"],
    "Hypothyroidism": ["Levothyroxine"],
    "Migraine": ["Sumatriptan", "Ibuprofen", "Acetaminophen"],
    "Tension Headache": ["Ibuprofen", "Acetaminophen", "Aspirin"]
}

FORBIDDEN_PAIRS = [
    ("Aspirin", "Warfarin"),
    ("Ibuprofen", "Warfarin"),
    ("Metformin", "Alcohol"),
    ("Beta_Blockers", "Verapamil"),
    ("ACE_Inhibitors", "Potassium_Supplements"),
]

DRUG_ALTERNATIVES = {
    "Aspirin": "Acetaminophen",
    "Ibuprofen": "Acetaminophen",
    "Warfarin": "Dabigatran",
    "Metformin": "Sitagliptin",
    "Beta_Blockers": "Amlodipine",
    "ACE_Inhibitors": "Amlodipine",
}
