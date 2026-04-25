import difflib
import re
import os

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage


model_name = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:7b")
llm_configuration = ChatOllama(
    model=model_name,
    temperature=0
)


def correct_paragraph(user_sentence):
    response = llm_configuration.invoke([
        SystemMessage(content="Исправь грамматические и пунктуационные ошибки. Верни только исправленный текст."),
        HumanMessage(content=user_sentence)
    ])

    return response.content


def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text)


def is_punctuation(token):
    return re.match(r"[^\w\s]", token) is not None


def get_changed_word(user_sentence, corrected_sentence):
    incorrect_changes = []
    correct_changes = []

    user_words = tokenize(user_sentence)
    corrected_words = tokenize(corrected_sentence)

    matcher = difflib.SequenceMatcher(None, user_words, corrected_words)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():

        if tag == "equal":
            for u, c in zip(user_words[i1:i2], corrected_words[j1:j2]):
                incorrect_changes.append(u)
                correct_changes.append(c)

        elif tag == "replace":
            user_part = user_words[i1:i2]
            correct_part = corrected_words[j1:j2]

            for i in range(max(len(user_part), len(correct_part))):
                u = user_part[i] if i < len(user_part) else "<NONE>"
                c = correct_part[i] if i < len(correct_part) else "<NONE>"

                if is_punctuation(u) and not is_punctuation(c):
                    incorrect_changes.append(u)
                    correct_changes.append("<DELETED>")
                    incorrect_changes.append("<ADDED>")
                    correct_changes.append(c)
                else:
                    incorrect_changes.append(u)
                    correct_changes.append(c)

        elif tag == "delete":
            for u in user_words[i1:i2]:
                incorrect_changes.append(u)
                correct_changes.append("<DELETED>")

        elif tag == "insert":
            for c in corrected_words[j1:j2]:
                incorrect_changes.append("<ADDED>")
                correct_changes.append(c)

    return incorrect_changes, correct_changes


def word_pair(incorrect_list, correct_list):
    return list(zip(incorrect_list, correct_list))


if __name__ == "__main__":
    user_sentence = "Привіт? мене звати Артем Бояр!!!"

    ai_sentence = correct_paragraph(user_sentence)

    incorrect_words, correct_words = get_changed_word(user_sentence, ai_sentence)

    print(f"Исходное предложение: {user_sentence}")
    print(f"Исправленное предложение: {ai_sentence}")
    print(f"Пары изменений: {word_pair(incorrect_words, correct_words)}")