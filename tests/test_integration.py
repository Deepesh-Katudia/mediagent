import pytest
from agents.orchestrator import OrchestratorAgent, State


@pytest.fixture
def agent():
    return OrchestratorAgent()


def test_full_flu_pipeline(agent):
    response = agent.process("I have fever, body aches, fatigue, and chills")
    assert agent.state == State.DONE
    assert "Flu" in response
    assert "Treatment" in response
    assert "Drug Safety" in response


def test_full_migraine_pipeline(agent):
    response = agent.process("I have headache, nausea, and light sensitivity")
    assert agent.state == State.DONE
    assert "Migraine" in response


def test_full_diabetes_pipeline(agent):
    response = agent.process("I have frequent urination, excessive thirst, and blurred vision")
    assert agent.state == State.DONE
    assert "Diabetes" in response or "Type 2" in response


def test_incremental_symptom_collection(agent):
    r1 = agent.process("I feel tired")
    assert agent.state == State.COLLECTING
    r2 = agent.process("also have weight gain and cold intolerance")
    assert agent.state == State.DONE
    assert "Hypothyroidism" in r2


def test_reset_and_new_session(agent):
    agent.process("I have fever, body aches, fatigue")
    assert agent.state == State.DONE
    agent.reset()
    assert agent.state == State.COLLECTING
    agent.process("I have headache, nausea, light sensitivity")
    assert agent.state == State.DONE


def test_explanation_contains_all_sections(agent):
    response = agent.process("I have fever, cough, chest pain, and shortness of breath")
    assert "Diagnosis" in response
    assert "Treatment Plan" in response
    assert "Drug Safety" in response


def test_full_diabetes_pipeline_natural_language(agent):
    """Regression: 'always thirsty' and 'vision is blurry' must be extracted."""
    response = agent.process(
        "I have frequent urination and I am always thirsty and my vision is blurry"
    )
    assert agent.state == State.DONE
    assert "Diabetes" in response or "Type 2" in response
