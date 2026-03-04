import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gensim.models import KeyedVectors
from groq import Groq
import os
from models.b2_predictor import B2PredictorModel
from data.tokenizer import sentence_preprocess_english
app = FastAPI(title="ML Linguo Service")

model_dir = os.getenv("MODEL_DIR", "/models")
ve_model = KeyedVectors.load(f"{model_dir}/crawl_fasttext.kv")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
try:
    predictor: B2PredictorModel = joblib.load("inference/b2_model.pkl")
except FileNotFoundError:
    predictor = B2PredictorModel()
    print("Модель еще не обучена. Endpoint будет работать только после обучения")


class PredictRequest(BaseModel):
    features: dict

class SimilarRequest(BaseModel):
    arr: list[str]
    topn: int = 10


class WordLevelRequest(BaseModel):
    word: str
    translation: str

class SentenceRequest(BaseModel):
    word: str
    translation: str
    level: str


# preprocess part
class EnglishPreprocessRequest(BaseModel):
    sentence: str

@app.post("/similar")
def similar(req: SimilarRequest):
    valid_words = [w for w in req.arr if w in ve_model]

    if not valid_words:
        return {"result": []}

    try:
        result = ve_model.most_similar(valid_words, topn=req.topn)

        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/word_level")
def word_level(req: WordLevelRequest):
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Ты — эксперт по лингвистике и системе уровней CEFR. "
                    f"Тебе дано слово на иностранном языке и его перевод. "
                    f"Определи уровень владения языком (по шкале CEFR), на котором это слово обычно изучается. "
                    f"Слово: {req.word} "
                    f"Перевод/значение: {req.translation} "
                    f"Ответь одним значением из списка: A1, A2, B1, B2, C1, C2. "
                    f"Никаких пояснений, только уровень."
                )
            }
        ],
        temperature=0,
        max_completion_tokens=10
    )

    return completion.choices[0].message.content.strip()

@app.post("/sentence")
def sentence(req: SentenceRequest):
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты генерируешь только одно английское предложение. "
                    "Без пояснений, без форматирования, без лишнего текста."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Слово: {req.word}\n"
                    f"Уровень: {req.level}\n"
                    f"Перевод: {req.translation}\n\n"
                    f"Напиши одно естественное предложение с этим словом. "
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

@app.post("/predict")
def predict(req: PredictRequest):
    if not predictor.feature_names:
        raise HTTPException(status_code=400, detail="Модель не обучена")

    try:
        df = pd.DataFrame([req.features])

        missing_cols = [c for c in predictor.feature_names if c not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Отсутствуют колонки: {missing_cols}")

        df = df[predictor.feature_names]

        pred = predictor.model.predict(df)[0]
        return {"prediction": int(pred)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/preprocess-en")
def preprocess_en(req: EnglishPreprocessRequest):
    return sentence_preprocess_english(req.sentence)