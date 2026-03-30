import os
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import json

load_dotenv()
client = ChatOllama(model=os.getenv("OLLAMA_MODEL_NAME"), temperature=0, num_predict=20, base_url="http://localhost:11434")

def llm_word_level(word: str, translation: str) -> str:
    response = client.invoke([
        SystemMessage(content=
                      "Ты определяешь уровень CEFR для английского слова. "
                      "Выбери ОДИН наиболее подходящий уровень: A1, A2, B1, B2, C1 или C2. "
                      "Отвечай строго в формате JSON: {\"level\": \"XX\"}.\n\n"
                      "Примеры:\n"
                      "cat / кошка → {\"level\": \"A1\"}\n"
                      "weather / погода → {\"level\": \"A2\"}\n"
                      "moreover / более того → {\"level\": \"B2\"}\n"
                      "ambiguity / неоднозначность → {\"level\": \"C1\"}\n"
                      "ephemeral / мимолётный → {\"level\": \"C2\"}\n"
                      ),
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