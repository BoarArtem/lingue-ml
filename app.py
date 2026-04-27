import joblib
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from gensim.models import Word2Vec
# from groq import Groq
import os
import nltk

from inference.topic_predictor import TopicPredictor
from models.b2_predictor import B2PredictorModel
from models.llm_sentence_generate import llm_sentence_generate
from models.llm_word_level import llm_word_level
from models.llm_correct_paragraph import correct_paragraph, get_changed_word, word_pair
from inference.spam_classification_inference import spam_or_ham
from data.tokenizer import (
    sentence_preprocess_english,
    sentence_preprocess_russian,
    sentence_preprocess_spanish,
    sentence_preprocess_france,
    sentence_preprocess_german,
    sentence_preprocess_chinese
)
from models.spam_classification_model import SpamClassificationModel

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model_dir = os.getenv("MODEL_DIR", "models")  # for docker testing/production
ve_model = Word2Vec.load(f"{model_dir}/word2vec.model")# - for my local testing
# ve_model = Word2Vec.load(f"{model_dir}/word2vec.model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpamClassificationModel(
    vocab_size=10000,
    embed_dim=128,
    hidden_size=256,
    num_layers=2,
).to(device)
model.load_state_dict(torch.load(f"{model_dir}/spam_classification_model.pth", map_location=device))
model.eval()

# client = Groq(api_key=os.getenv("OPENAI_KEY"))

try:
    topic_predictor = TopicPredictor()
except Exception as e:
    print(f"Ошибка загрузки TopicPredictor: {e}")
    topic_predictor = None

try:
    predictor: B2PredictorModel = joblib.load(f"{model_dir}/b2_model.pkl")
except FileNotFoundError:
    predictor = B2PredictorModel()
    print("Модель еще не обучена")


class TopicRequest(BaseModel):
    sentences: list[str] = Field(
        ...,
        description="Список предложений для определения темы",
        example=["I love coding in Python", "The football match was intense"]
    )


class SingleTopicRequest(BaseModel):
    sentence: str = Field(
        ...,
        description="Одно предложение для определения темы",
        example="I love coding in Python"
    )


class PredictRequest(BaseModel):
    features: dict = Field(
        ...,
        example={
            'unique_words': 1500,
            'words_a1': 600,
            'words_a2': 500,
            'words_b1': 400,
            'words_b2': 0,
            'avg_acc_7d': 0.88,
            'avg_acc_30d': 0.85,
            'avg_time_sec': 6.0,
            'words_day_7d': 30,
            'words_day_30d': 900,
            'streak': 20,
            'sessions_week': 14
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

class SpamClassificationRequest(BaseModel):
    user_sentence: str = Field(example="sex sex drugs drugs gun")


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
    "/predict_topic",  # Обрати внимание, тут в единственном числе
    tags=["Machine Learning"],
    summary="Определение темы для одного предложения",
    description="Принимает одну строку и возвращает предсказанную тему.",
    response_description="Предсказанная тема"
)
def predict_topic(req: SingleTopicRequest):
    if not topic_predictor:
        raise HTTPException(status_code=500, detail="Topic model is not initialized")

    try:

        result = topic_predictor.get_topic(req.sentence)
        return {"topic": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict_topics",
    tags=["Machine Learning"],
    summary="Определение темы текста",
    description="Принимает массив строк и возвращает предсказанные темы для каждой из них.",
    response_description="Массив предсказанных тем"
)
def predict_topics(req: TopicRequest):
    if not topic_predictor:
        raise HTTPException(status_code=500, detail="Topic model is not initialized")

    try:
        results = topic_predictor.get_topics(req.sentences)
        return {"topics": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    user_sentence = req.user_sentence
    ai_sentence = correct_paragraph(user_sentence)

    original_user = user_sentence
    original_ai = ai_sentence

    incorrect_words, correct_words = get_changed_word(user_sentence, ai_sentence)

    return {
        "User sentence": original_user,
        "AI sentence": original_ai,
        "Changing pair": word_pair(incorrect_words, correct_words)
    }

@app.post(
    "/spam_classification",
    tags=["Machine Learning"],
    summary="Классификация карточки на спам",
    description="ИИ перед тем как пользователь выложит свою колоду, проходить по каждой карточке и даёт метку (spam | ham), дальше все карточки с пометкой спам удаляют из колоды и оставляют только ham",
    response_description="Метка класса: spam or ham"
)
def spam_classification(req: SpamClassificationRequest):
    return {'class': spam_or_ham(req.user_sentence)}