import boto3
import os

MODEL_DIR = os.getenv("MODEL_DIR", "/models")
os.makedirs(MODEL_DIR, exist_ok=True)

s3 = boto3.client(
    service_name="s3",
    endpoint_url=os.getenv("R2_ENDPOINT"),
    aws_access_key_id=os.getenv("R2_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
)

files = [
    "crawl_fasttext.kv",
    "crawl-300d-2M.vec",
    "crawl_fasttext.kv.vectors.npy"
]


for file in files:
    path = f"{MODEL_DIR}/{file}"

    if not os.path.exists(path):
        print(f"Downloading {file}...")
        s3.download_file("ml-models", file, path)

print("All models downloaded")

