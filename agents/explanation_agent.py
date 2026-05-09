"""
ExplanationAgent — Plain-English Diagnostic Narrative Formatter
===============================================================

AI Technique: Template-driven natural language generation (NLG)

Role in pipeline:
    The ExplanationAgent is the final specialist invoked by the
    OrchestratorAgent.  It receives the accumulated outputs of every upstream
    agent — the collected symptoms, ranked diagnoses, A*-planned treatment
    pathway, and CSP drug safety report — and composes them into a single,
    coherent, human-readable response for display in the chat interface.

    The agent performs no additional reasoning of its own; its sole
    responsibility is *presentation*.  By isolating formatting logic here, the
    rest of the pipeline remains decoupled from any specific output format,
    making it straightforward to swap in alternative renderers (e.g. JSON,
    HTML, speech) without touching the reasoning agents.

    Output structure:
        1. Primary diagnosis with blended confidence percentage.
        2. List of matched symptoms that contributed to the diagnosis.
        3. Numbered treatment pathway with human-readable step labels.
        4. Drug safety status — either a clear-all confirmation or a warning
           block listing each interaction and recommended alternatives.
        5. Secondary differential diagnoses (if any) with their confidence
           percentages.
"""


class ExplanationAgent:
    """Natural-language formatter that synthesises all agent outputs into a report.

    The agent is entirely stateless — ``explain()`` is a pure function of its
    arguments and produces no side effects.  A single instance can therefore
    be shared safely across multiple sessions.
    """

    def explain(self, symptoms: list, diagnosis: list, treatment: list, safety: dict) -> str:
        """Format the full diagnostic consultation result as a Markdown-flavoured string.

        Assembles a structured report in the following order:

        1. **Diagnosis line** — Top-ranked disease name and blended confidence
           expressed as a percentage.
        2. **Matched symptoms** — Union of required and supporting symptom
           matches from the FOL inference pass.
        3. **Treatment Plan** — Numbered list of treatment steps, with
           internal node identifiers translated to user-friendly labels via
           ``step_labels``.  Unknown node names are passed through unchanged.
        4. **Drug Safety** — If safe, lists the prescribed drugs.  If unsafe,
           lists each violation and any available drug alternatives.
        5. **Differential diagnoses** — All non-primary candidates with their
           confidence percentages, displayed as a bulleted list.

        Parameters
        ----------
        symptoms : list[str]
            Canonical symptom tokens collected during the session (used
            implicitly — currently displayed upstream in the chat history
            rather than repeated here).
        diagnosis : list[dict]
            Ranked list of diagnosis dicts as returned by
            ``DiagnosisAgent.diagnose()``.  Each dict must contain at least
            ``"disease"``, ``"confidence"``, ``"matched_required"``, and
            ``"matched_supporting"`` keys.
        treatment : list[str]
            Ordered list of treatment node names as returned by
            ``TreatmentPlannerAgent.plan_treatment()``.
        safety : dict
            Drug safety report as returned by ``DrugSafetyAgent.check_safety()``,
            containing ``"safe"``, ``"drugs"``, ``"violations"``, and
            ``"alternatives"`` keys.

        Returns
        -------
        str
            A multi-line Markdown-flavoured string suitable for direct
            rendering in the MediAgent chat interface.  Returns a fallback
            message advising the user to consult a doctor when ``diagnosis``
            is empty.
        """
        if not diagnosis:
            return "I was unable to identify a condition from the symptoms provided. Please consult a doctor."

        top = diagnosis[0]
        confidence_pct = int(top["confidence"] * 100)
        disease = top["disease"]
        matched = top.get("matched_required", []) + top.get("matched_supporting", [])

        lines = [
            f"**Diagnosis:** {disease} ({confidence_pct}% confidence)",
            f"**Matched symptoms:** {', '.join(matched) if matched else 'N/A'}",
            "",
            f"**Treatment Plan:**",
        ]

        step_labels = {
            "rest_prescribed": "Rest and hydration",
            "otc_medication": "Over-the-counter medication",
            "prescription_medication": "Prescription medication required",
            "specialist_referral": "Specialist referral recommended",
            "hospitalization": "Hospitalization may be needed",
            "treated": "Recovery expected",
        }
        for i, step in enumerate(treatment, 1):
            lines.append(f"  {i}. {step_labels.get(step, step)}")

        lines.append("")
        drugs = safety.get("drugs", [])
        if safety.get("safe"):
            lines.append(f"**Drug Safety:** Safe to use — {', '.join(drugs) if drugs else 'no drugs prescribed'}")
        else:
            lines.append("**Drug Safety:** ⚠ Warning — interaction detected!")
            for v in safety.get("violations", []):
                lines.append(f"  - {v}")
            alts = safety.get("alternatives", {})
            if alts:
                alt_text = ", ".join(f"{k} → {v}" for k, v in alts.items())
                lines.append(f"  Alternatives: {alt_text}")

        if len(diagnosis) > 1:
            lines.append("")
            lines.append("**Other possibilities considered:**")
            for alt in diagnosis[1:]:
                lines.append(f"  - {alt['disease']} ({int(alt['confidence'] * 100)}%)")

        return "\n".join(lines)
