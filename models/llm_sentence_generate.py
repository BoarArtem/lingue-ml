import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def llm_sentence_generate(word: str, level: str, language: str) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты генерируешь только одно предложение на нужном языке заданый в промпте. "
                    "Без пояснений, без форматирования, без лишнего текста."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Слово: {word}\n"
                    f"Уровень: {level}\n"
                    f"Язык на котором надо придумать предложение: {language}"
                    f"Напиши одно естественное предложение с этим словом. Строго на языке которое я написал"
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
    print(llm_sentence_generate("Пельмени", "B1", "Russian"))
