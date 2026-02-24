import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def llm_sentence_generate(word: str, translation: str, level: str) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY_ARTEM"))

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты генерируешь только одно английское предложение. "
                    "Без пояснений, без форматирования, без лишнего текста."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Слово: {word}\n"
                    f"Уровень: {level}\n"
                    f"Перевод: {translation}\n\n"
                    f"Напиши одно естественное предложение с этим словом. "
                    f"Ответ — только предложение."
                )
            }
        ],
        temperature=0.4,
        max_completion_tokens=40,
        top_p=1,
        stream=False
    )

    result = completion.choices[0].message.content.strip()

    result = result.split("\n")[0]
    if "." in result:
        result = result[:result.find(".") + 1]

    return result.strip()


if __name__ == "__main__":
    print(llm_sentence_generate("unfortunately", "к сожалению", "B1"))