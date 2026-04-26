import boto3
from boto3.s3.transfer import TransferConfig
import os
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("R2_ENDPOINT"),
    aws_access_key_id=os.getenv("R2_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
    region_name="auto"
)

config = TransferConfig(
    multipart_threshold=50 * 1024 * 1024,
    multipart_chunksize=50 * 1024 * 1024,
    max_concurrency=4,
    use_threads=True
)

files_to_upload = [
    "../data/tokens_cache.pkl",
    "../data/datasets/spam_Emails_data.csv"
]

for file in files_to_upload:
    filename = os.path.basename(file)

    print(f"Uploading {file}...")

    s3.upload_file(
        Filename=file,
        Bucket="ml-models",
        Key=filename,
        Config=config
    )

print("Upload complete")