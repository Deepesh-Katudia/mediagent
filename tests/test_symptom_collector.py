import pytest
from agents.symptom_collector import SymptomCollectionAgent


@pytest.fixture
def agent():
    return SymptomCollectionAgent()


def test_extract_canonical_symptom(agent):
    result = agent.extract_symptoms("I have a fever and cough")
    assert "fever" in result
    assert "cough" in result


def test_extract_synonym_normalizes_to_canonical(agent):
    result = agent.extract_symptoms("I have a high temperature and I feel tired")
    assert "fever" in result
    assert "fatigue" in result


def test_extract_multiword_synonym(agent):
    result = agent.extract_symptoms("I have chest pain and shortness of breath")
    assert "chest_pain" in result
    assert "shortness_of_breath" in result


def test_extract_returns_no_duplicates(agent):
    result = agent.extract_symptoms("fever and high temperature")
    assert result.count("fever") == 1


def test_extract_empty_text_returns_empty(agent):
    result = agent.extract_symptoms("")
    assert result == []


def test_extract_unrecognized_text_returns_empty(agent):
    result = agent.extract_symptoms("I feel fantastic today")
    assert result == []


def test_has_enough_symptoms_true(agent):
    assert agent.has_enough_symptoms(["fever", "cough"]) is True


def test_has_enough_symptoms_false(agent):
    assert agent.has_enough_symptoms(["fever"]) is False


def test_has_enough_symptoms_empty(agent):
    assert agent.has_enough_symptoms([]) is False


def test_get_followup_question_no_symptoms(agent):
    q = agent.get_followup_question([])
    assert "symptom" in q.lower() or "experiencing" in q.lower()


def test_get_followup_question_with_symptoms(agent):
    q = agent.get_followup_question(["fever", "cough"])
    assert "fever" in q
    assert "cough" in q


def test_extract_always_thirsty_maps_to_excessive_thirst(agent):
    result = agent.extract_symptoms("I am always thirsty")
    assert "excessive_thirst" in result


def test_extract_vision_is_blurry_maps_to_blurred_vision(agent):
    result = agent.extract_symptoms("my vision is blurry")
    assert "blurred_vision" in result


def test_extract_dizzy_maps_to_dizziness(agent):
    result = agent.extract_symptoms("I feel dizzy and have a headache")
    assert "dizziness" in result
    assert "headache" in result


def test_extract_diabetes_full_natural_sentence(agent):
    result = agent.extract_symptoms(
        "I have frequent urination and I am always thirsty and my vision is blurry"
    )
    assert "frequent_urination" in result
    assert "excessive_thirst" in result
    assert "blurred_vision" in result


def test_extract_stressed_maps_to_fatigue(agent):
    result = agent.extract_symptoms("I am feeling stressed and have a headache")
    assert "fatigue" in result
    assert "headache" in result


def test_extract_headche_typo_maps_to_headache(agent):
    result = agent.extract_symptoms("I have headche and fever")
    assert "headache" in result
    assert "fever" in result
