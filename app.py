from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gensim.models import KeyedVectors
from groq import Groq
import os

app = FastAPI(title="ML Linguo Service")

ve_model = KeyedVectors.load("inference/crawl_fasttext.kv")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class SimilarRequest(BaseModel):
    arr: list[str]
    topn: int = 10


class WordLevelRequest(BaseModel):
    word: str
    translation: str

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