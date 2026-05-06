import pytest
from agents.drug_safety_agent import DrugSafetyAgent


@pytest.fixture
def agent():
    return DrugSafetyAgent()


def test_check_returns_dict(agent):
    result = agent.check_safety("Flu")
    assert isinstance(result, dict)


def test_check_safe_disease_has_safe_true(agent):
    result = agent.check_safety("Hypothyroidism")
    assert result["safe"] is True


def test_check_result_contains_drugs_key(agent):
    result = agent.check_safety("Flu")
    assert "drugs" in result


def test_check_result_contains_violations_key(agent):
    result = agent.check_safety("Flu")
    assert "violations" in result


def test_check_result_contains_alternatives_key(agent):
    result = agent.check_safety("Flu")
    assert "alternatives" in result


def test_check_angina_with_warfarin_flags_aspirin(agent):
    result = agent.check_safety("Angina", existing_drugs=["Warfarin"])
    assert result["safe"] is False
    assert len(result["violations"]) > 0


def test_check_no_existing_drugs_is_safe_for_simple_disease(agent):
    result = agent.check_safety("Common Cold")
    assert result["safe"] is True


def test_check_violations_include_drug_names(agent):
    result = agent.check_safety("Angina", existing_drugs=["Warfarin"])
    violation_text = str(result["violations"])
    assert "Aspirin" in violation_text or "Warfarin" in violation_text


def test_check_alternatives_provided_when_violation(agent):
    result = agent.check_safety("Angina", existing_drugs=["Warfarin"])
    if not result["safe"]:
        assert len(result["alternatives"]) > 0
