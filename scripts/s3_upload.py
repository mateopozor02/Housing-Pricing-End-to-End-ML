"""
s3_upload.py

Script to upload key datasets and model artifacts to AWS S3.

Usage:
    uv run python scripts/s3_upload.py --bucket <bucket-name> [--prefix <s3-folder>]

Uploads:
- data/processed/feature_engineered_holdout_data.csv
- data/processed/feature_engineered_train_data.csv
- data/processed/cleaning_holdout_data.csv
- models/lightgbm_best_model.pkl

Requires AWS credentials to be configured (via environment or ~/.aws/credentials).
"""

import argparse
from pathlib import Path
import boto3
import sys

# Files to upload
FILES = [
    Path("data/processed/feature_engineered_holdout_data.csv"),
    Path("data/processed/feature_engineered_train_data.csv"),
    Path("data/processed/cleaning_holdout_data.csv"),
    Path("models/lightgbm_best_model.pkl"),
]


def upload_file(s3_client, file_path, bucket, s3_key):
    print(f"Uploading {file_path} to s3://{bucket}/{s3_key}")
    s3_client.upload_file(str(file_path), bucket, s3_key)


def main():
    parser = argparse.ArgumentParser(description="Upload datasets and model to AWS S3.")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--prefix", default="", help="S3 folder prefix (optional)")
    args = parser.parse_args()

    s3 = boto3.client("s3")

    for file_path in FILES:
        if not file_path.exists():
            print(f"Warning: {file_path} does not exist, skipping.", file=sys.stderr)
            continue
        s3_key = f"{args.prefix.strip('/')}/" if args.prefix else ""
        s3_key += file_path.as_posix()
        upload_file(s3, file_path, args.bucket, s3_key)

    print("Upload complete.")


if __name__ == "__main__":
    main()
