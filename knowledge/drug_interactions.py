"""
drug_interactions.py
====================
Knowledge base for drug safety checking via Constraint Satisfaction
(Russell & Norvig, Ch. 5).

This module defines:
- DISEASE_DRUGS: maps each supported disease to its standard first-line
  medications. DrugSafetyAgent uses this to assemble a candidate drug list
  from the top diagnoses returned by DiagnosisAgent.
- FORBIDDEN_PAIRS: binary constraints representing clinically dangerous
  drug-drug (or drug-substance) combinations. DrugSafetyAgent enforces
  these as hard CSP constraints and eliminates any violating medication.
- DRUG_ALTERNATIVES: fallback substitutions offered to the patient when a
  preferred drug is removed due to a forbidden-pair violation.

AI technique supported: Constraint Satisfaction Problem (DrugSafetyAgent).
"""

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
