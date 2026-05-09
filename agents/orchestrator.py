"""
OrchestratorAgent — MediAgent Pipeline Controller
===================================================

AI Technique: Finite State Machine (FSM)

Role in pipeline:
    The OrchestratorAgent is the top-level controller for the MediAgent system.
    It manages a deterministic, six-state finite state machine that sequences
    every other specialist agent in a well-defined order:

        COLLECTING → DIAGNOSING → PLANNING → CHECKING → EXPLAINING → DONE

    On each call to ``process()``, the orchestrator advances through as many
    states as possible in a single turn, or pauses and returns a clarifying
    question to the user when more information is required (e.g., too few
    symptoms collected, or diagnostic confidence is below threshold).

    The FSM design guarantees that downstream agents (diagnosis, treatment,
    drug-safety, explanation) are invoked only after their preconditions are
    satisfied, preventing partial or inconsistent results from being surfaced
    to the user.
"""

from enum import Enum, auto
from agents.symptom_collector import SymptomCollectionAgent
from agents.diagnosis_agent import DiagnosisAgent
from agents.treatment_planner import TreatmentPlannerAgent
from agents.drug_safety_agent import DrugSafetyAgent
from agents.explanation_agent import ExplanationAgent

CONFIDENCE_THRESHOLD = 0.5


class State(Enum):
    """Enumeration of the six ordered states in the diagnostic FSM.

    States progress linearly from COLLECTING through to DONE.  The only
    allowed backward transition is from DIAGNOSING back to COLLECTING when
    diagnostic confidence is insufficient.
    """

    COLLECTING = auto()
    DIAGNOSING = auto()
    PLANNING = auto()
    CHECKING = auto()
    EXPLAINING = auto()
    DONE = auto()


class OrchestratorAgent:
    """Top-level controller that drives the MediAgent diagnostic pipeline.

    The OrchestratorAgent instantiates and coordinates five specialist agents:

    * :class:`~agents.symptom_collector.SymptomCollectionAgent` — NLP-based
      symptom extraction and follow-up question generation.
    * :class:`~agents.diagnosis_agent.DiagnosisAgent` — FOL + Naive Bayes
      blended inference to rank candidate diagnoses.
    * :class:`~agents.treatment_planner.TreatmentPlannerAgent` — A* search
      over a treatment graph to produce an ordered care pathway.
    * :class:`~agents.drug_safety_agent.DrugSafetyAgent` — CSP-based drug
      interaction checking with alternative suggestions.
    * :class:`~agents.explanation_agent.ExplanationAgent` — Plain-English
      narrative formatter for final user output.

    Public interface:
        ``process(user_input)`` — accepts free-text patient input and returns
        either a follow-up question or the full diagnostic report depending on
        the current FSM state.

    Session lifecycle:
        Call ``reset()`` to clear accumulated state between consultations.
    """

    def __init__(self):
        """Instantiate all specialist agents and initialise session state."""
        self._symptom_agent = SymptomCollectionAgent()
        self._diagnosis_agent = DiagnosisAgent()
        self._treatment_agent = TreatmentPlannerAgent()
        self._drug_agent = DrugSafetyAgent()
        self._explanation_agent = ExplanationAgent()
        self.reset()

    def reset(self):
        """Reset the FSM to its initial state and clear all accumulated data.

        Sets the current state back to ``COLLECTING`` and empties the
        symptom list, diagnosis results, treatment pathway, and safety report.
        Call this between independent patient consultations to avoid state
        bleed-over from a previous session.
        """
        self.state = State.COLLECTING
        self.symptoms: list = []
        self.diagnosis: list = []
        self.treatment: list = []
        self.safety: dict = {}

    def process(self, user_input: str) -> str:
        """Advance the FSM by one or more states and return the next response.

        This is the single entry point for all patient interaction.  On each
        invocation the method attempts to progress through every pending FSM
        state in sequence, stopping only when user input is required or the
        session reaches ``DONE``.

        State transitions:

        * **COLLECTING** — Extracts symptoms from ``user_input`` via the
          SymptomCollectionAgent.  If the collected symptom count is below the
          minimum threshold, returns a follow-up question and stays in this
          state.  Otherwise advances to DIAGNOSING.

        * **DIAGNOSING** — Runs FOL + ML inference.  If no diagnosis exceeds
          ``CONFIDENCE_THRESHOLD``, reverts to COLLECTING and requests more
          detail.  Otherwise advances to PLANNING.

        * **PLANNING** — Runs A* search on the treatment graph for the
          top-ranked disease.  Advances to CHECKING.

        * **CHECKING** — Runs CSP-based drug safety check for the top-ranked
          disease.  Advances to EXPLAINING.

        * **EXPLAINING** — Formats and returns the full diagnostic narrative.
          Advances to DONE.

        * **DONE** — Returns a session-complete message.

        Parameters
        ----------
        user_input : str
            Raw free-text message from the patient.

        Returns
        -------
        str
            Either a clarifying follow-up question (when more information is
            needed) or the complete diagnostic report (when the pipeline has
            run to completion).
        """
        if self.state == State.DONE:
            return "Session complete. Type 'new' to start over."

        if self.state == State.COLLECTING:
            new_symptoms = self._symptom_agent.extract_symptoms(user_input)
            self.symptoms = sorted(set(self.symptoms + new_symptoms))

            if not self._symptom_agent.has_enough_symptoms(self.symptoms):
                return self._symptom_agent.get_followup_question(self.symptoms)

            self.state = State.DIAGNOSING

        if self.state == State.DIAGNOSING:
            self.diagnosis = self._diagnosis_agent.diagnose(self.symptoms)

            if not self.diagnosis or self.diagnosis[0]["confidence"] < CONFIDENCE_THRESHOLD:
                self.state = State.COLLECTING
                return (
                    f"I have your symptoms ({', '.join(self.symptoms)}) but need more detail. "
                    "Can you describe any additional symptoms?"
                )

            self.state = State.PLANNING

        if self.state == State.PLANNING:
            top_disease = self.diagnosis[0]["disease"]
            self.treatment = self._treatment_agent.plan_treatment(top_disease)
            self.state = State.CHECKING

        if self.state == State.CHECKING:
            top_disease = self.diagnosis[0]["disease"]
            self.safety = self._drug_agent.check_safety(top_disease)
            self.state = State.EXPLAINING

        if self.state == State.EXPLAINING:
            explanation = self._explanation_agent.explain(
                self.symptoms, self.diagnosis, self.treatment, self.safety
            )
            self.state = State.DONE
            return explanation

        return "Processing..."
