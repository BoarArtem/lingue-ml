import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from gensim.models import KeyedVectors
from groq import Groq
import os

from models.b2_predictor import B2PredictorModel
from data.tokenizer import (
    sentence_preprocess_english,
    sentence_preprocess_russian,
    sentence_preprocess_spanish,
    sentence_preprocess_france,
    sentence_preprocess_german,
    sentence_preprocess_chinese
)

app = FastAPI(
    title="ML Linguo Service",
    description="""
ML сервис для задач Linguo.

Возможности API:

• поиск похожих слов (FastText embeddings)  
• определение уровня слова (CEFR)  
• генерация предложений  
• ML предсказания  
• preprocessing текста  

Все ответы возвращаются в JSON.
""",
    version="1.0.0"
)

model_dir = os.getenv("MODEL_DIR", "/models")
ve_model = KeyedVectors.load(f"{model_dir}/crawl_fasttext.kv")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

try:
    predictor: B2PredictorModel = joblib.load("inference/b2_model.pkl")
except FileNotFoundError:
    predictor = B2PredictorModel()
    print("Модель еще не обучена")


class PredictRequest(BaseModel):
    features: dict = Field(
        ...,
        example={
            "emails_sent": 10,
            "open_rate": 0.42,
            "click_rate": 0.11
        }
    )


class SimilarRequest(BaseModel):
    arr: list[str] = Field(
        ...,
        description="Список слов для поиска похожих",
        example=["dog", "cat"]
    )

    topn: int = Field(
        default=10,
        description="Количество похожих слов",
        example=5
    )


class WordLevelRequest(BaseModel):
    word: str = Field(example="nevertheless")
    translation: str = Field(example="тем не менее")


class SentenceRequest(BaseModel):
    word: str = Field(example="dog")
    translation: str = Field(example="собака")
    level: str = Field(example="A1")
    language: str = Field(example="en")


class PreprocessRequest(BaseModel):
    sentence: str = Field(example="Dogs are running in the park")
    language: str = Field(example="en")


# =========================
# ENDPOINTS
# =========================

@app.post(
    "/similar",
    tags=["Embeddings"],
    summary="Поиск похожих слов",
    description="""
Возвращает список слов, наиболее похожих на переданные.

Используется **FastText модель** (`gensim KeyedVectors`).

Алгоритм:

1. Проверяются слова из `arr`, которые есть в словаре.
2. Вызывается `most_similar`.
3. Возвращается список `(word, similarity)`.

Similarity — косинусное сходство.
""",
    response_description="Список похожих слов"
)
def similar(req: SimilarRequest):
    valid_words = [w for w in req.arr if w in ve_model]

    if not valid_words:
        return {"result": []}

    try:
        result = ve_model.most_similar(valid_words, topn=req.topn)
        return {"result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/word_level",
    tags=["LLM"],
    summary="Определить уровень слова CEFR",
    description="""
Определяет уровень сложности слова по шкале **CEFR**.

Используется LLM модель через **Groq API**.

Модель анализирует:

- слово
- перевод

И возвращает один уровень:

A1, A2, B1, B2, C1, C2

Ответ всегда строка без пояснений.
""",
    response_description="Уровень CEFR"
)
def word_level(req: WordLevelRequest):

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Ты — эксперт по лингвистике и системе уровней CEFR. "
                    f"Тебе дано слово и перевод. "
                    f"Определи уровень CEFR.\n"
                    f"Слово: {req.word}\n"
                    f"Перевод: {req.translation}\n"
                    f"Ответ: A1, A2, B1, B2, C1 или C2."
                )
            }
        ],
        temperature=0,
        max_completion_tokens=10
    )

    return completion.choices[0].message.content.strip()


@app.post(
    "/sentence",
    tags=["LLM"],
    summary="Сгенерировать предложение",
    description="""
Генерирует одно естественное предложение с заданным словом.

Параметры:

- `word` — слово
- `translation` — перевод
- `level` — уровень CEFR
- `language` — язык предложения

Ограничения:

• одно предложение  
• без объяснений  
• только текст
""",
    response_description="Сгенерированное предложение"
)
def sentence(req: SentenceRequest):

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "Ты генерируешь только одно предложение."
            },
            {
                "role": "user",
                "content": (
                    f"Слово: {req.word}\n"
                    f"Уровень: {req.level}\n"
                    f"Перевод: {req.translation}\n"
                    f"Язык: {req.language}\n"
                    f"Напиши одно предложение."
                )
            }
        ],
        temperature=0.4,
        max_completion_tokens=40
    )

    result = completion.choices[0].message.content.strip()

    result = result.split("\n")[0]
    if "." in result:
        result = result[:result.find(".") + 1]

    return result.strip()


@app.post(
    "/predict",
    tags=["Machine Learning"],
    summary="ML предсказание",
    description="""
Использует обученную ML модель `B2PredictorModel`.

Шаги:

1. принимаются признаки `features`
2. создаётся pandas DataFrame
3. проверяются необходимые колонки
4. вызывается `model.predict`

Если модель не обучена — возвращается ошибка.
""",
    response_description="Результат предсказания"
)
def predict(req: PredictRequest):

    if not predictor.feature_names:
        raise HTTPException(status_code=400, detail="Модель не обучена")

    try:
        df = pd.DataFrame([req.features])

        missing_cols = [c for c in predictor.feature_names if c not in df.columns]

        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Отсутствуют колонки: {missing_cols}"
            )

        df = df[predictor.feature_names]

        pred = predictor.model.predict(df)[0]

        return {"prediction": int(pred)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/preprocess",
    tags=["NLP"],
    summary="Предобработка предложения",
    description="""
Нормализует предложение.

Возможные операции:

• токенизация  
• очистка текста  
• лемматизация  

Поддерживаемые языки:

- en — English
- ru — Russian
- es — Spanish
- fr — French
- de — German
- ch — Chinese
""",
    response_description="Токены предложения"
)
def preprocess(req: PreprocessRequest):

    if req.language == "en":
        return sentence_preprocess_english(req.sentence)

    if req.language == "ru":
        return sentence_preprocess_russian(req.sentence)

    if req.language == "es":
        return sentence_preprocess_spanish(req.sentence)

    if req.language == "fr":
        return sentence_preprocess_france(req.sentence)

    if req.language == "de":
        return sentence_preprocess_german(req.sentence)

    if req.language == "ch":
        return sentence_preprocess_chinese(req.sentence)