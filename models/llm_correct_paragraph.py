import os
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()
client = ChatOllama(model=os.getenv("OLLAMA_MODEL_NAME"), temperature=0)

def correct_paragraph(user_sentence):
    response = client.invoke([
        SystemMessage(content=(
            "You are a grammar corrector.",
            "Correct any grammatical errors and adjust verb tenses according to context.",
            "Do not change the meaning or words unnecessarily.",
            "Do not translate.",
            "Do not add explanations, parentheses, or extra text.",
            "Always respond in the same language as the input.",
            "Return ONLY the corrected sentence, nothing else."
        )),
        HumanMessage(content=user_sentence),
    ])
    return response.content

# if __name__ == "__main__":
#     print(correct_paragraph("Я ел суп завтра"))