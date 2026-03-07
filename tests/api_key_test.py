import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    api_key = os.environ.get("GROQ_API_KEY")

def test_api_key_loaded():
    assert api_key is not None, "API-key: GROQ_API_KEY не найдена"
    assert api_key != "", "API-key: GROQ_API_KEY пуста"