import os
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import json

load_dotenv()
client = ChatOllama(model=os.getenv("OLLAMA_MODEL_NAME"), temperature=0, num_predict=20)

def llm_word_level(word: str, translation: str) -> str:
    response = client.invoke([
        SystemMessage(content=[f"Ты определяешь уровень CEFR слова. \nОтвечай строго JSON: {{'level':'A1/A2/B1/B2/C1/C2'}}",
                      "Ответ должен быть подан без будь каких символов, кроме - {'level': 'A1/A2/B1/B2/C1/C2'}"]),
        HumanMessage(content=f"Слово: {word}\nПеревод: {translation}")
    ])

    text = response.content

    try:
        data = json.loads(text.replace("'", '"'))
        return data.get("level", "Unknown")
    except json.JSONDecodeError:
        return "Unknown"