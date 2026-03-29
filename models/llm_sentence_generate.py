import os
from openai import OpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()
client = ChatOllama(model="gemma3:1b", temperature=0)

def llm_sentence_generate(word: str, level: str, language: str) -> str:
    response = client.invoke([
            SystemMessage(content=[
                "Ты генерируешь только одно предложение на нужном указанном языке пользователем, ",
                "и с уровнем сложности, который будет предложено пользователем, ",
                "без пояснений, без форматирования, без лишнего текста."]),

            HumanMessage(content=[
                    f"Слово: {word}\n"
                    f"Уровень: {level}\n"
                    f"Язык: {language}\n"
                    "Напиши одно естественное предложение с этим словом. "
                    "Ответ — строго только одно предложение."]),
        ],
    )

    result = response.content

    result = result.split("\n")[0]

    if "." in result:
        result = result[:result.find(".") + 1]


    return result.strip()