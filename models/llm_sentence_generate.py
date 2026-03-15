import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

def llm_sentence_generate(word: str, level: str, language: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты генерируешь только одно предложение на нужном указанном языке пользователем, "
                    "без пояснений, без форматирования, без лишнего текста."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Слово: {word}\n"
                    f"Уровень: {level}\n"
                    f"Язык: {language}\n"
                    "Напиши одно естественное предложение с этим словом. "
                    "Ответ — строго только одно предложение."
                )
            }
        ],
        temperature=0.4,
        max_completion_tokens=40,
        top_p=1
    )

    result = response.choices[0].message.content.strip()

    result = result.split("\n")[0]

    if "." in result:
        result = result[:result.find(".") + 1]


    return result.strip()
