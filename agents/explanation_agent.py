class ExplanationAgent:

    def explain(self, symptoms: list, diagnosis: list, treatment: list, safety: dict) -> str:
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
