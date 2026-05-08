import os
import sys
import pytest
import openai
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.openai_voice import (
    CHAT_MODEL,
    DOCTOR_SYSTEM_PROMPT,
    consult_gpt,
    get_openai_client,
)


# ---------------------------------------------------------------------------
# get_openai_client
# ---------------------------------------------------------------------------


def test_get_openai_client_raises_if_key_missing():
    # Arrange: remove OPENAI_API_KEY from environment
    with patch.dict(os.environ, {}, clear=True):
        with patch("utils.openai_voice.load_dotenv"):
            # Act / Assert
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                get_openai_client()


def test_get_openai_client_returns_client():
    # Arrange: provide a fake key in the environment
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-fake-key"}):
        with patch("utils.openai_voice.load_dotenv"):
            # Act
            client = get_openai_client()
            # Assert
            assert isinstance(client, openai.OpenAI)


# ---------------------------------------------------------------------------
# consult_gpt helpers
# ---------------------------------------------------------------------------


def _make_chat_mock(reply_text: str):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = reply_text
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


# ---------------------------------------------------------------------------
# consult_gpt
# ---------------------------------------------------------------------------


def test_consult_gpt_system_prompt_is_first():
    # Arrange
    mock_client = _make_chat_mock("Some reply.")
    history = []

    # Act
    consult_gpt(mock_client, "I feel sick", history)

    # Assert
    call_kwargs = mock_client.chat.completions.create.call_args
    messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0]
    assert messages[0]["role"] == "system"
    assert DOCTOR_SYSTEM_PROMPT in messages[0]["content"]


def test_consult_gpt_uses_chat_model():
    # Arrange
    mock_client = _make_chat_mock("Some reply.")
    history = []

    # Act
    consult_gpt(mock_client, "I feel sick", history)

    # Assert
    call_kwargs = mock_client.chat.completions.create.call_args
    model_used = call_kwargs.kwargs.get("model") or call_kwargs.args[0]
    assert model_used == CHAT_MODEL
    assert model_used == "gpt-4o-mini"


def test_consult_gpt_uses_max_completion_tokens():
    # Arrange
    mock_client = _make_chat_mock("Some reply.")
    history = []

    # Act
    consult_gpt(mock_client, "I feel sick", history)

    # Assert
    call_kwargs = mock_client.chat.completions.create.call_args
    assert call_kwargs.kwargs.get("max_completion_tokens") == 200


def test_consult_gpt_appends_to_history():
    # Arrange
    mock_client = _make_chat_mock("You should rest.")
    history = []

    # Act
    consult_gpt(mock_client, "I feel sick", history)

    # Assert: one user turn + one assistant turn appended
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_consult_gpt_preserves_prior_history():
    # Arrange
    mock_client = _make_chat_mock("Drink more water.")
    history = [
        {"role": "user", "content": "I have a fever"},
        {"role": "assistant", "content": "Stay hydrated."},
    ]

    # Act
    consult_gpt(mock_client, "What else should I do?", history)

    # Assert: two existing + two new turns
    assert len(history) == 4


def test_consult_gpt_returns_reply():
    # Arrange
    mock_client = _make_chat_mock("You should rest.")
    history = []

    # Act
    result = consult_gpt(mock_client, "I feel tired", history)

    # Assert
    assert result == "You should rest."


def test_consult_gpt_raises_if_no_choices():
    # Arrange
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = []
    mock_client.chat.completions.create.return_value = mock_response
    history = []

    # Act / Assert
    with pytest.raises(ValueError, match="No completion"):
        consult_gpt(mock_client, "I feel sick", history)


def test_consult_gpt_raises_on_api_error():
    # Arrange
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = openai.APIStatusError(
        "insufficient_quota", response=MagicMock(status_code=429), body={}
    )

    # Act / Assert
    with pytest.raises(openai.APIStatusError):
        consult_gpt(mock_client, "I have a headache", [])
