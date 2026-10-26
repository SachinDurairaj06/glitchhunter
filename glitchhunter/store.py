"""Amazon S3 artifact store for hunt outputs (trace, reports, evidence images)."""

from __future__ import annotations

import mimetypes
from pathlib import Path


def upload_dir(local: Path, bucket: str, prefix: str) -> str:
    """Upload every file under `local` to s3://bucket/prefix/ (server-side encrypted)."""
    import boto3

    s3 = boto3.client("s3")
    for path in sorted(local.rglob("*")):
        if not path.is_file():
            continue
        key = f"{prefix.rstrip('/')}/{path.relative_to(local).as_posix()}"
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        s3.upload_file(str(path), bucket, key,
                       ExtraArgs={"ContentType": ctype, "ServerSideEncryption": "AES256"})
    return f"s3://{bucket}/{prefix.rstrip('/')}/"


def presigned_url(bucket: str, key: str, expires: int = 3600) -> str:
    import boto3

    return boto3.client("s3").generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires)
