import pytest
from agents.explanation_agent import ExplanationAgent


@pytest.fixture
def agent():
    return ExplanationAgent()


@pytest.fixture
def sample_diagnosis():
    return [
        {
            "disease": "Flu",
            "confidence": 0.82,
            "fol_score": 0.85,
            "ml_prob": 0.77,
            "matched_required": ["fever", "body_aches", "fatigue"],
            "matched_supporting": ["chills"]
        }
    ]


@pytest.fixture
def sample_treatment():
    return ["rest_prescribed", "otc_medication", "treated"]


@pytest.fixture
def sample_safety():
    return {"safe": True, "drugs": ["Ibuprofen", "Acetaminophen"], "violations": [], "alternatives": {}}


def test_explain_returns_string(agent, sample_diagnosis, sample_treatment, sample_safety):
    result = agent.explain(["fever", "fatigue"], sample_diagnosis, sample_treatment, sample_safety)
    assert isinstance(result, str)


def test_explain_contains_disease_name(agent, sample_diagnosis, sample_treatment, sample_safety):
    result = agent.explain(["fever", "fatigue"], sample_diagnosis, sample_treatment, sample_safety)
    assert "Flu" in result


def test_explain_contains_confidence(agent, sample_diagnosis, sample_treatment, sample_safety):
    result = agent.explain(["fever", "fatigue"], sample_diagnosis, sample_treatment, sample_safety)
    assert "82" in result or "0.82" in result


def test_explain_contains_treatment_step(agent, sample_diagnosis, sample_treatment, sample_safety):
    result = agent.explain(["fever", "fatigue"], sample_diagnosis, sample_treatment, sample_safety)
    assert "rest" in result.lower() or "otc" in result.lower() or "medication" in result.lower()


def test_explain_safe_drugs_mentioned(agent, sample_diagnosis, sample_treatment, sample_safety):
    result = agent.explain(["fever", "fatigue"], sample_diagnosis, sample_treatment, sample_safety)
    assert "safe" in result.lower() or "ibuprofen" in result.lower()


def test_explain_unsafe_shows_warning(agent, sample_diagnosis, sample_treatment):
    unsafe_safety = {
        "safe": False,
        "drugs": ["Aspirin", "Warfarin"],
        "violations": ["Aspirin + Warfarin interaction detected"],
        "alternatives": {"Aspirin": "Acetaminophen"}
    }
    result = agent.explain(["fever", "fatigue"], sample_diagnosis, sample_treatment, unsafe_safety)
    assert "warning" in result.lower() or "interaction" in result.lower() or "⚠" in result


def test_explain_empty_diagnosis_returns_message(agent, sample_treatment, sample_safety):
    result = agent.explain(["fever"], [], sample_treatment, sample_safety)
    assert "not" in result.lower() or "unable" in result.lower() or "insufficient" in result.lower()
