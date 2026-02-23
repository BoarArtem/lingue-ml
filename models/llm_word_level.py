import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def llm_world_level(word: str, translation: str) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY_ARTEM"))

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Ты — эксперт по лингвистике и системе уровней CEFR. "
                    f"Тебе дано слово на иностранном языке и его перевод. "
                    f"Определи уровень владения языком (по шкале CEFR), на котором это слово обычно изучается. "
                    f"Слово: {word} "
                    f"Перевод/значение: {translation} "
                    f"Ответь одним значением из списка: A1, A2, B1, B2, C1, C2. "
                    f"Никаких пояснений, только уровень."
                )
            }
        ],
        temperature=1,
        max_completion_tokens=10,
        top_p=1,
        stream=True,
        stop=None
    )

    result = ""
    for chunk in completion:
        result += chunk.choices[0].delta.content or ""

    return result.strip()


if __name__ == "__main__":
    level = llm_world_level("unfortunately", "к сожалению")
    print(level)