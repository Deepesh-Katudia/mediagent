import pytest
from agents.diagnosis_agent import DiagnosisAgent


@pytest.fixture
def agent():
    return DiagnosisAgent()


def test_fol_returns_list(agent):
    result = agent.run_fol_inference(["fever", "body_aches", "fatigue"])
    assert isinstance(result, list)


def test_fol_detects_flu_from_required_symptoms(agent):
    result = agent.run_fol_inference(["fever", "body_aches", "fatigue"])
    diseases = [r["disease"] for r in result]
    assert "Flu" in diseases


def test_fol_returns_max_three_results(agent):
    result = agent.run_fol_inference(["fever", "headache", "cough", "chest_pain", "fatigue"])
    assert len(result) <= 3


def test_fol_results_sorted_by_score_descending(agent):
    result = agent.run_fol_inference(["fever", "body_aches", "fatigue", "chills", "headache"])
    scores = [r["fol_score"] for r in result]
    assert scores == sorted(scores, reverse=True)


def test_fol_no_match_returns_empty(agent):
    result = agent.run_fol_inference(["indigestion"])
    assert result == []


def test_fol_result_has_required_keys(agent):
    result = agent.run_fol_inference(["fever", "body_aches", "fatigue"])
    assert len(result) > 0
    keys = result[0].keys()
    assert "disease" in keys
    assert "fol_score" in keys
    assert "matched_required" in keys
    assert "matched_supporting" in keys


def test_fol_migraine_detected(agent):
    result = agent.run_fol_inference(["headache", "nausea", "light_sensitivity"])
    diseases = [r["disease"] for r in result]
    assert "Migraine" in diseases


def test_fol_no_required_symptoms_not_returned(agent):
    result = agent.run_fol_inference(["cough"])
    if "Bronchitis" in [r["disease"] for r in result]:
        bronchitis = next(r for r in result if r["disease"] == "Bronchitis")
        assert bronchitis["fol_score"] < 1.0


def test_diagnose_returns_blended_confidence(agent):
    result = agent.diagnose(["fever", "body_aches", "fatigue"])
    assert len(result) > 0
    assert "confidence" in result[0]
    assert 0.0 <= result[0]["confidence"] <= 1.0
