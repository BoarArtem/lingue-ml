import os
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import json

load_dotenv()
client = ChatOllama(model=os.getenv("OLLAMA_MODEL_NAME"), temperature=0, num_predict=20)

def llm_word_level(word: str, translation: str) -> str:
    response = client.invoke([
        SystemMessage(content=[
            "Ты определяешь уровень CEFR для английского слова. "
            "Выбери ОДИН наиболее подходящий уровень: A1, A2, B1, B2, C1 или C2. "
            "Отвечай строго в формате JSON с одним полем. "
            'Пример правильного ответа: {"level": "B1"}. '
            "Запрещено указывать несколько уровней через слэш или запятую. "
            "Только один уровень в поле level."
        ]),
        HumanMessage(content=f"Слово: {word}\nПеревод: {translation}")
    ])

    text = response.content
    if text.startswith("```json"):
        text = text.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(text.replace("'", '"'))
        return data.get("level", "Unknown")
    except json.JSONDecodeError:
        return "Unknown"