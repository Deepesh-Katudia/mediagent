from enum import Enum, auto
from agents.symptom_collector import SymptomCollectionAgent
from agents.diagnosis_agent import DiagnosisAgent
from agents.treatment_planner import TreatmentPlannerAgent
from agents.drug_safety_agent import DrugSafetyAgent
from agents.explanation_agent import ExplanationAgent

CONFIDENCE_THRESHOLD = 0.5


class State(Enum):
    COLLECTING = auto()
    DIAGNOSING = auto()
    PLANNING = auto()
    CHECKING = auto()
    EXPLAINING = auto()
    DONE = auto()


class OrchestratorAgent:

    def __init__(self):
        self._symptom_agent = SymptomCollectionAgent()
        self._diagnosis_agent = DiagnosisAgent()
        self._treatment_agent = TreatmentPlannerAgent()
        self._drug_agent = DrugSafetyAgent()
        self._explanation_agent = ExplanationAgent()
        self.reset()

    def reset(self):
        self.state = State.COLLECTING
        self.symptoms: list = []
        self.diagnosis: list = []
        self.treatment: list = []
        self.safety: dict = {}

    def process(self, user_input: str) -> str:
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
