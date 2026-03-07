import os
from dotenv import load_dotenv

load_dotenv()

def test_api_key_loaded():
    api_key = os.getenv("GROQ_API_KEY")

    assert api_key is not None, "Groq API key not loaded"
    assert api_key != "", "Groq API key is empty"