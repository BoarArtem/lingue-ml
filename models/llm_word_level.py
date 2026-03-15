import os
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

def llm_word_level(word: str, translation: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ты определяешь уровень CEFR слова."},
            {"role": "user", "content": f"Слово: {word}\nПеревод: {translation}\nОтвечай строго JSON: {{'level':'A1/A2/B1/B2/C1/C2'}}"}
        ],
        max_completion_tokens=20
    )

    text = response.choices[0].message.content.strip()
    try:
        data = json.loads(text.replace("'", '"'))
        return data.get("level", "Unknown")
    except json.JSONDecodeError:
        return "Unknown"

