import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from gensim.models import Word2Vec
from groq import Groq
import os
import nltk
from models.llm_correct_paragraph import correct_paragraph, get_changed_word 
from inference.context_evaluator import evaluate_context_with_groq, calculate_fsrs
from models.b2_predictor import B2PredictorModel
from models.llm_sentence_generate import llm_sentence_generate
from models.llm_word_level import llm_word_level
from models.llm_correct_paragraph import correct_paragraph, get_changed_word
from data.tokenizer import (
    sentence_preprocess_english,
    sentence_preprocess_russian,
    sentence_preprocess_spanish,
    sentence_preprocess_france,
    sentence_preprocess_german,
    sentence_preprocess_chinese
)

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

app = FastAPI(
    title="ML Linguo Service",
    description="""
ML сервис для Linguo.

Возможности API:

• поиск похожих слов (FastText embeddings)  
• определение уровня слова (CEFR)  
• генерация предложений  
• ML предсказания  
• preprocessing текста  
""",
    version="v2.9.3"
)

model_dir = os.getenv("MODEL_DIR", "models")

try:
    ve_model = Word2Vec.load(f"{model_dir}/word2vec.model")
except FileNotFoundError:
    ve_model = None
    print("Внимание: word2vec.model не найден локально. Эндпоинт /similar временно недоступен.")

client = Groq(api_key=os.getenv("OPENAI_KEY"))


try:
    predictor: B2PredictorModel = joblib.load(f"{model_dir}/b2_model.pkl")
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
    level: str = Field(example="A1")
    language: str = Field(example="en | English")


class PreprocessRequest(BaseModel):
    sentence: str = Field(example="Dogs are running in the park")
    language: str = Field(example="en")

class CorrectParagraphRequest(BaseModel):
    user_sentence: str = Field(example="I ate pizza yesterday")

class FSRSEvaluationRequest(BaseModel):
    target_phrase: str
    user_sentence: str
    expected_level: str = "A1" 

class FSRSEvaluationResponse(BaseModel):
    fsrs_grade: int
    is_used: bool
    fits_context: bool
    sentence_level: str
    grammar_errors: int
    corrected_sentence: str


@app.post("/evaluate_sentence", response_model=FSRSEvaluationResponse)
def evaluate_sentence(req: FSRSEvaluationRequest):
    try:
        
        corrected_sentence = correct_paragraph(req.user_sentence)
        correct_words, incorrect_words = get_changed_word(req.user_sentence, corrected_sentence)
        grammar_errors_count = len(incorrect_words)

        
        context_data = evaluate_context_with_groq(client, req.target_phrase, req.user_sentence)

        
        fsrs_grade = calculate_fsrs(context_data, grammar_errors_count, req.expected_level)

        
        return FSRSEvaluationResponse(
            fsrs_grade=fsrs_grade,
            is_used=context_data.get("is_used", False),
            fits_context=context_data.get("fits_context", False),
            sentence_level=context_data.get("sentence_level", "A1"),
            grammar_errors=grammar_errors_count,
            corrected_sentence=corrected_sentence
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/similar",
    tags=["Embeddings"],
    summary="Поиск похожих слов",
    description="""
Возвращает список слов, наиболее похожих на переданные.

Используется **FastText модель** (`gensim KeyedVectors`).
""",
    response_description="Список похожих слов"
)
def similar(req: SimilarRequest):
    result = ve_model.wv.most_similar(req.arr, topn=req.topn)

    return result

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
    result = llm_word_level(
        req.word,
        req.translation
    )

    return result


@app.post(
    "/sentence",
    tags=["LLM"],
    summary="Сгенерировать предложение",
    description="""
Генерирует одно естественное предложение с заданным словом.

Параметры:

- `word` — слово
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
    result = llm_sentence_generate(
        req.word,
        req.level,
        req.language
    )

    return result

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


@app.post(
    "/correct_paragraph",
    tags=["LLM"],
    summary="Исправление ошибок в предложение пользователя",
    description="""
    Пользователь при создании карточки может с помощью ИИ проверить на правильность написания предложения (грамматика или пунктуация)
    """,
    response_description="Объект в котором возвращаеться исправленое предложение, массив правильных слов которое написало ИИ и массив неправильных слов с ошибками или пунктуация"
)
def correct_paragraph_checking(req: CorrectParagraphRequest):
    correct_sentence = correct_paragraph(req.user_sentence)

    correct_words, incorrect_words = get_changed_word(req.user_sentence, correct_sentence)

    ai_checking = {
        "correction": correct_sentence,
        "corrected_word": correct_words,
        "incorrected_word": incorrect_words
    }

    return ai_checking
