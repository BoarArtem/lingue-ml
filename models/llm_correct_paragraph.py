import os
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import difflib
import re

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

def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text)

def get_changed_word(user_sentence, corrected_sentence):
    correct_changes = []
    incorrect_changes = []

    user_words = tokenize(user_sentence)
    corrected_words = tokenize(corrected_sentence)

    diff = difflib.ndiff(user_words, corrected_words)

    for d in diff:
        token = d[2:]

        if d.startswith("+ "):
            correct_changes.append(token)
        elif d.startswith("- "):
            # if re.fullmatch(r"[^\w\s]", token):
            incorrect_changes.append(token)

    return correct_changes, incorrect_changes



if __name__ == "__main__":
    user_sentence = "Я! завтра сьел кашу"
    correct_sentence = correct_paragraph(user_sentence)

    correct_words, incorrect_words = get_changed_word(user_sentence, correct_sentence)

    ai_checking = {
        "correction": correct_sentence,
        "corrected_word": correct_words,
        "incorrected_word": incorrect_words
    }

    print(ai_checking)
