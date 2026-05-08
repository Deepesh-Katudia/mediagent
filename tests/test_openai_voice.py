import os
import sys
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.openai_voice import (
    CHAT_MODEL,
    DOCTOR_SYSTEM_PROMPT,
    TTS_MODEL,
    TTS_VOICE,
    WHISPER_MODEL,
    consult_gpt,
    get_openai_client,
    run_voice_turn,
    synthesize_speech,
    transcribe_audio,
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
            import openai

            client = get_openai_client()
            # Assert
            assert isinstance(client, openai.OpenAI)


# ---------------------------------------------------------------------------
# transcribe_audio
# ---------------------------------------------------------------------------


def test_transcribe_audio_calls_whisper():
    # Arrange
    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.return_value = MagicMock(text="hello")
    audio_bytes = b"fake-audio-data"

    # Act
    transcribe_audio(mock_client, audio_bytes)

    # Assert
    call_kwargs = mock_client.audio.transcriptions.create.call_args
    assert call_kwargs.kwargs.get("model") == WHISPER_MODEL or (
        call_kwargs.args and call_kwargs.args[0] == WHISPER_MODEL
    )


def test_transcribe_audio_file_name_is_wav():
    # Arrange
    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.return_value = MagicMock(text="hello")
    audio_bytes = b"fake-audio-data"

    # Act
    transcribe_audio(mock_client, audio_bytes)

    # Assert: inspect the `file` argument passed to create
    call_kwargs = mock_client.audio.transcriptions.create.call_args
    file_obj = call_kwargs.kwargs.get("file") or call_kwargs.args[1]
    assert file_obj.name == "audio.wav"


def test_transcribe_audio_returns_text():
    # Arrange
    mock_client = MagicMock()
    mock_client.audio.transcriptions.create.return_value = MagicMock(
        text="I have a headache"
    )
    audio_bytes = b"fake-audio-data"

    # Act
    result = transcribe_audio(mock_client, audio_bytes)

    # Assert
    assert result == "I have a headache"


# ---------------------------------------------------------------------------
# consult_gpt
# ---------------------------------------------------------------------------


def _make_chat_mock(reply_text: str):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = reply_text
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


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


# ---------------------------------------------------------------------------
# synthesize_speech
# ---------------------------------------------------------------------------


def test_synthesize_speech_uses_tts1():
    # Arrange
    mock_client = MagicMock()
    mock_client.audio.speech.create.return_value = MagicMock(read=lambda: b"mp3data")

    # Act
    synthesize_speech(mock_client, "Take rest.")

    # Assert
    call_kwargs = mock_client.audio.speech.create.call_args
    assert call_kwargs.kwargs.get("model") == TTS_MODEL
    assert call_kwargs.kwargs.get("model") == "tts-1"


def test_synthesize_speech_uses_nova_voice():
    # Arrange
    mock_client = MagicMock()
    mock_client.audio.speech.create.return_value = MagicMock(read=lambda: b"mp3data")

    # Act
    synthesize_speech(mock_client, "Take rest.")

    # Assert
    call_kwargs = mock_client.audio.speech.create.call_args
    assert call_kwargs.kwargs.get("voice") == TTS_VOICE
    assert call_kwargs.kwargs.get("voice") == "nova"


def test_synthesize_speech_returns_bytes():
    # Arrange
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.read.return_value = b"mp3data"
    mock_client.audio.speech.create.return_value = mock_response

    # Act
    result = synthesize_speech(mock_client, "Take rest.")

    # Assert
    assert result == b"mp3data"


# ---------------------------------------------------------------------------
# run_voice_turn
# ---------------------------------------------------------------------------


def test_run_voice_turn_returns_triple():
    # Arrange
    mock_client = MagicMock()

    mock_client.audio.transcriptions.create.return_value = MagicMock(
        text="I have a headache"
    )

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Take paracetamol."
    mock_client.chat.completions.create.return_value = mock_response

    mock_speech = MagicMock()
    mock_speech.read.return_value = b"mp3data"
    mock_client.audio.speech.create.return_value = mock_speech

    history = []

    # Act
    result = run_voice_turn(mock_client, b"fake-audio", history)

    # Assert: must be a 3-tuple of (str, str, bytes)
    assert isinstance(result, tuple)
    assert len(result) == 3
    transcript, reply, audio = result
    assert isinstance(transcript, str)
    assert isinstance(reply, str)
    assert isinstance(audio, bytes)


def test_run_voice_turn_updates_history():
    # Arrange
    mock_client = MagicMock()

    mock_client.audio.transcriptions.create.return_value = MagicMock(
        text="I have a headache"
    )

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Take paracetamol."
    mock_client.chat.completions.create.return_value = mock_response

    mock_speech = MagicMock()
    mock_speech.read.return_value = b"mp3data"
    mock_client.audio.speech.create.return_value = mock_speech

    history = []

    # Act
    run_voice_turn(mock_client, b"fake-audio", history)

    # Assert: consult_gpt appends one user + one assistant turn
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
