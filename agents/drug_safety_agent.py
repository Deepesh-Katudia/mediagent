from constraint import Problem
from knowledge.drug_interactions import DISEASE_DRUGS, FORBIDDEN_PAIRS, DRUG_ALTERNATIVES


class DrugSafetyAgent:

    def check_safety(self, disease: str, existing_drugs: list = None) -> dict:
        disease_drugs = DISEASE_DRUGS.get(disease, [])
        all_drugs = list(set(disease_drugs + (existing_drugs or [])))

        if not all_drugs:
            return {"safe": True, "drugs": [], "violations": [], "alternatives": {}}

        problem = Problem()
        for drug in all_drugs:
            problem.addVariable(drug, ["safe", "flagged"])

        for drug_a, drug_b in FORBIDDEN_PAIRS:
            if drug_a in all_drugs and drug_b in all_drugs:
                problem.addConstraint(
                    lambda a, b: not (a == "safe" and b == "safe"),
                    (drug_a, drug_b)
                )

        solutions = problem.getSolutions()

        violations = []
        alternatives = {}
        for drug_a, drug_b in FORBIDDEN_PAIRS:
            if drug_a in all_drugs and drug_b in all_drugs:
                violations.append(f"{drug_a} + {drug_b} interaction detected")
                for d in (drug_a, drug_b):
                    if d in DRUG_ALTERNATIVES:
                        alternatives[d] = DRUG_ALTERNATIVES[d]

        is_safe = len(violations) == 0

        return {
            "safe": is_safe,
            "drugs": all_drugs,
            "violations": violations,
            "alternatives": alternatives,
        }
