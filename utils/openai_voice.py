import os
from dotenv import load_dotenv
import openai

CHAT_MODEL = "gpt-4o-mini"

DOCTOR_SYSTEM_PROMPT = (
    "You are an experienced, compassionate medical doctor providing a voice consultation. "
    "Give clear, concise answers in 2-3 sentences. Focus on likely causes and whether "
    "the patient should seek immediate care. Always end with: 'Remember, I am an AI assistant "
    "and this is not a substitute for professional medical advice.'"
)


def get_openai_client() -> openai.OpenAI:
    load_dotenv()
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")
    return openai.OpenAI(api_key=key)


def consult_gpt(client: openai.OpenAI, user_text: str, history: list) -> str:
    messages = [{"role": "system", "content": DOCTOR_SYSTEM_PROMPT}] + history + \
               [{"role": "user", "content": user_text}]
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        max_completion_tokens=200,
    )
    if not response.choices:
        raise ValueError("No completion returned by API")
    reply = response.choices[0].message.content
    # Mutates caller's list intentionally — voice_history in app.py is the same object
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply})
    return reply
