import pytest
from agents.orchestrator import OrchestratorAgent, State


@pytest.fixture
def agent():
    return OrchestratorAgent()


def test_initial_state_is_collecting(agent):
    assert agent.state == State.COLLECTING


def test_process_returns_string(agent):
    result = agent.process("I have fever and body aches and fatigue")
    assert isinstance(result, str)


def test_collecting_state_asks_for_symptoms(agent):
    result = agent.process("hello")
    assert "symptom" in result.lower() or "experiencing" in result.lower()


def test_sufficient_symptoms_triggers_diagnosis(agent):
    agent.process("I have fever and body aches and fatigue")
    assert agent.state in (State.DIAGNOSING, State.PLANNING, State.CHECKING, State.EXPLAINING, State.DONE)


def test_low_confidence_loops_back_to_collecting(agent):
    agent.state = State.DIAGNOSING
    agent.symptoms = ["fever"]
    result = agent.process("just fever")
    assert "more" in result.lower() or "symptom" in result.lower() or agent.state == State.COLLECTING


def test_full_pipeline_reaches_done(agent):
    agent.process("I have fever, body aches, fatigue, and chills")
    while agent.state != State.DONE:
        response = agent.process("continue")
        if agent.state == State.DONE:
            break
    assert agent.state == State.DONE


def test_reset_clears_state(agent):
    agent.process("I have fever and body aches")
    agent.reset()
    assert agent.state == State.COLLECTING
    assert agent.symptoms == []
    assert agent.diagnosis == []


def test_process_after_done_prompts_reset(agent):
    agent.state = State.DONE
    result = agent.process("anything")
    assert "new" in result.lower() or "reset" in result.lower() or "start" in result.lower()
