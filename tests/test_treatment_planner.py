import pytest
from agents.treatment_planner import TreatmentPlannerAgent


@pytest.fixture
def agent():
    return TreatmentPlannerAgent()


def test_plan_returns_list(agent):
    result = agent.plan_treatment("Flu")
    assert isinstance(result, list)


def test_plan_flu_starts_with_rest(agent):
    result = agent.plan_treatment("Flu")
    assert result[0] == "rest_prescribed"


def test_plan_ends_with_treated(agent):
    result = agent.plan_treatment("Flu")
    assert result[-1] == "treated"


def test_plan_pneumonia_starts_with_prescription(agent):
    result = agent.plan_treatment("Pneumonia")
    assert result[0] == "prescription_medication"


def test_plan_severe_disease_not_starting_with_rest(agent):
    result = agent.plan_treatment("Angina")
    assert result[0] != "rest_prescribed"


def test_plan_unknown_disease_returns_default_path(agent):
    result = agent.plan_treatment("Unknown Disease")
    assert result[-1] == "treated"
    assert len(result) >= 2


def test_plan_path_nodes_are_valid_states(agent):
    from knowledge.treatment_graph import TREATMENT_STATES
    result = agent.plan_treatment("Migraine")
    for node in result:
        assert node in TREATMENT_STATES


def test_plan_result_has_no_duplicate_consecutive_nodes(agent):
    result = agent.plan_treatment("Hypertension")
    for i in range(len(result) - 1):
        assert result[i] != result[i + 1]
