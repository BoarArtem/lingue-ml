import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import difflib
import re

load_dotenv()
client = ChatOpenAI(model=os.getenv("OPENAI_MODEL_NAME"), temperature=0)

def correct_paragraph(user_sentence):
    response = client.invoke([
        SystemMessage(content="""Ты — ассистент для исправления грамматических и пунктуационных ошибок в тексте.

Твоя задача:
1. Исправить все грамматические ошибки (орфография, согласование слов, падежи, времена глаголов и т.д.)
2. Исправить все пунктуационные ошибки (запятые, точки, тире, кавычки и т.д.)
3. Вернуть ТОЛЬКО исправленный текст, без комментариев и объяснений
4. Сохранить исходный стиль и тон сообщения
5. Не изменять смысл предложения

Правила:
- Если текст уже корректен, верни его без изменений
- Не добавляй лишних слов или пояснений
- Не переписывай предложения полностью — исправляй только ошибки
- Сохраняй форматирование (переносы строк, списки и т.д.)

Примеры:

Пользователь: "Я пошол в магазин и купил молако"
Ты: "Я пошёл в магазин и купил молоко"

Пользователь: "Привет как дела"
Ты: "Привет, как дела?"

Пользователь: "Мне нравится этот фильм потому что он интересный"
Ты: "Мне нравится этот фильм, потому что он интересный"

Если написано на одном языке, на том и исправляй.
В случае если в одном предложении два языка сразу - то посмотри где больше слов, и где будет больше на том и меняй
"""
        ),
        HumanMessage(content=user_sentence),
    ])

    return response.content

def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text)

def is_punctuation(token):
    return re.match(r"[^\w\s]", token)

def get_changed_word(user_sentence, corrected_sentence):
    incorrect_changes = []
    correct_changes = []

    user_words = tokenize(user_sentence)
    corrected_words = tokenize(corrected_sentence)

    matcher = difflib.SequenceMatcher(None, user_words, corrected_words)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():

        if tag == "replace":
            user_part = user_words[i1:i2]
            correct_part = corrected_words[j1:j2]

            for u, c in zip(user_part, correct_part):

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

def check_len(user_sentence_arr, ai_sentence_arr):
    if len(user_sentence_arr) == len(ai_sentence_arr):
        return 'Good'
    else:
        return 'Not good'

def word_pair(incorrect_list, correct_list):
    return list(zip(incorrect_list, correct_list))

# if __name__ == "__main__":
#     user_sentence = "Я лублу кушать вишну и яблко"
#     ai_sentence = correct_paragraph(user_sentence)
#
#     tokenize_user = tokenize(user_sentence)
#     tokenize_ai = tokenize(ai_sentence)
#
#     print("User: ", tokenize_user)
#     print("AI: ", tokenize_ai)
#
#     # print(check_len(tokenize_user, tokenize_ai))


if __name__ == "__main__":
    user_sentence = "Привіт? мене звати Артем Бояр!!!"
    ai_sentence = correct_paragraph(user_sentence)

    original_user = user_sentence
    original_ai = ai_sentence

    incorrect_words, correct_words = get_changed_word(user_sentence, ai_sentence)

    print(f"Исходное предложение: {original_user}")
    print(f"Исправленное предложение: {original_ai}")
    print(f"Пары изменений: {word_pair(incorrect_words, correct_words)}")