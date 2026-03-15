import os
from dotenv import load_dotenv

load_dotenv()
# groq_api_key = os.getenv("GROQ_API_KEY")
openai_api_key = os.getenv("OPENAI_KEY")
r2_endpoint = os.getenv("R2_ENDPOINT")
r2_access_key = os.getenv("R2_ACCESS_KEY")
r2_secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY")

if not openai_api_key:
    openai_api_key = os.environ.get("OPENAI_KEY")
if not r2_endpoint:
    r2_endpoint = os.environ.get("R2_ENDPOINT")
if not r2_access_key:
    r2_access_key = os.environ.get("R2_ACCESS_KEY")
if not r2_secret_access_key:
    r2_secret_access_key = os.environ.get("R2_SECRET_ACCESS_KEY")

def test_openai_api_key_loaded():
    assert openai_api_key is not None, "API-key: OPENAI_KEY не найдена"
    assert openai_api_key != "", "API-key: OPENAI_KEY пуста"

def test_r2_endpoint_loaded():
    assert r2_endpoint is not None, "API-key: R2_ENDPOINT не найдена"
    assert r2_endpoint != "", "API-key: R2_ENDPOINT пуста"

def test_r2_access_key_loaded():
    assert r2_access_key is not None, "API-key: R2_ACCESS_KEY не найдена"
    assert r2_access_key != "", "API-key: R2_ACCESS_KEY пуста"

def test_r2_secret_access_key():
    assert r2_secret_access_key is not None, "API-key: R2_SECRET_ACCESS_KEY не найдена"
    assert r2_secret_access_key != "", "API-key: R2_SECRET_ACCESS_KEY пуста"