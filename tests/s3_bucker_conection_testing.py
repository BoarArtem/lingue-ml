import boto3
import boto3
import os
from dotenv import load_dotenv
load_dotenv()

MODEL_DIR = os.getenv("MODEL_DIR", "/models")
os.makedirs(MODEL_DIR, exist_ok=True)

print("ACCESS:", os.getenv("R2_ACCESS_KEY"))
print("SECRET:", os.getenv("R2_SECRET_ACCESS_KEY"))
print("ENDPOINT:", os.getenv("R2_ENDPOINT"))

s3 = boto3.client(
    service_name="s3",
    endpoint_url=os.getenv("R2_ENDPOINT"),
    aws_access_key_id=os.getenv("R2_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
)