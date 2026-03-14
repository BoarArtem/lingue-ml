import os
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def llm_world_level(word: str, translation: str) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты определяешь уровень CEFR слова. "
                    "Отвечай только одним значением: A1, A2, B1, B2, C1 или C2. "
                    "Никаких других слов."
                )
            },
            {
                "role": "user",
                "content": f"Слово: {word}\nПеревод: {translation}\nУровень CEFR:"
            }
        ],
        temperature=0,
        max_completion_tokens=5
    )

    result = completion.choices[0].message.content.strip()

    match = re.search(r"(A1|A2|B1|B2|C1|C2)", result)
    return match.group(1) if match else "Unknown"