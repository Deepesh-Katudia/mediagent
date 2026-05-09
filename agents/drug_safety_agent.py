"""
DrugSafetyAgent — Constraint Satisfaction Problem (CSP) Drug Interaction Checker
=================================================================================

AI Technique: Constraint Satisfaction Problem (CSP) solving via the
``python-constraint`` library

Role in pipeline:
    The DrugSafetyAgent is invoked by the OrchestratorAgent after the
    TreatmentPlannerAgent has produced a care pathway.  Its purpose is to
    verify that the drugs associated with the top-ranked diagnosis do not
    present dangerous pairwise interactions, and to suggest safer alternatives
    when conflicts are detected.

    CSP formulation:
        * **Variables** — Each drug in the combined set (disease-standard drugs
          plus any pre-existing patient medications) is modelled as a CSP
          variable with domain ``{"safe", "flagged"}``.
        * **Constraints** — For every pair ``(drug_a, drug_b)`` listed in
          ``FORBIDDEN_PAIRS``, a binary constraint is added stating that both
          drugs cannot simultaneously hold the value ``"safe"``.  This encodes
          the clinical rule: "these two drugs must not both be prescribed
          together."
        * **Solution** — The CSP solver enumerates all valid assignments.
          However, the primary output consumed by downstream agents is not the
          solution set itself but the *violation list*: any forbidden pair
          where both drugs are present in the active drug set is recorded as
          an interaction, regardless of the CSP assignment, because the
          clinical risk exists as long as both drugs could be co-prescribed.

    Alternative suggestions:
        For each flagged drug, ``DRUG_ALTERNATIVES`` is consulted to surface a
        recommended substitution that avoids the interaction.
"""

from constraint import Problem
from knowledge.drug_interactions import DISEASE_DRUGS, FORBIDDEN_PAIRS, DRUG_ALTERNATIVES


class DrugSafetyAgent:
    """CSP-based agent that checks drug interaction safety for a given disease.

    The agent is stateless; each call to ``check_safety`` constructs and
    solves an independent CSP instance scoped to the drugs relevant to that
    call.  This design allows the agent to be safely shared across concurrent
    sessions without locking.
    """

    def check_safety(self, disease: str, existing_drugs: list = None) -> dict:
        """Check for dangerous drug interactions using a CSP formulation.

        Constructs a constraint satisfaction problem where each drug is a
        variable and forbidden pairings are binary inequality constraints.
        Identifies all pairs of co-present drugs that violate interaction
        rules, collects their safer alternatives, and returns a structured
        safety report.

        Parameters
        ----------
        disease : str
            Canonical disease identifier used to look up the standard drug
            regimen from ``DISEASE_DRUGS``.
        existing_drugs : list[str], optional
            Pre-existing medications the patient is already taking.  These are
            merged with the disease-specific drugs before constraint checking
            to catch cross-regimen interactions.  Defaults to an empty list.

        Returns
        -------
        dict
            A safety report with the following keys:

            ``safe`` : bool
                ``True`` if no forbidden drug pairs are co-present.
            ``drugs`` : list[str]
                All drug names included in the analysis.
            ``violations`` : list[str]
                Human-readable descriptions of each detected interaction
                (e.g. ``"aspirin + warfarin interaction detected"``).
            ``alternatives`` : dict[str, str]
                Mapping of each flagged drug to a recommended alternative drug
                name, sourced from ``DRUG_ALTERNATIVES``.
        """
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
