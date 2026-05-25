"""S3 publish helpers.

Bucket layout:
    /ping                            — JSON {"success": true}, content-type application/json
    /api/libraries                   — libraries response
    /api/libraries/main/items        — items list
    /api/libraries/main/filterdata   — empty filterdata
    /api/libraries/main/collections  — empty collections
    /api/libraries/main/search       — empty search
    /api/items/<id>                  — item detail
    /books/<id>/audio.m4b            — m4b audio
    /books/<id>/cover.jpg            — cover art

The `/api/items/<id>/cover` and `/api/items/<id>/download` URLs are rewritten
by the CloudFront Function to `/books/<id>/cover.jpg` and `/books/<id>/audio.m4b`
respectively, so we do not store anything at those paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import boto3


class Publisher:
    def __init__(self, bucket: str):
        self.bucket = bucket
        self.s3 = boto3.client("s3")

    def put_json(self, key: str, body: dict[str, Any]) -> None:
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(body).encode("utf-8"),
            ContentType="application/json",
            CacheControl="no-store",
        )

    def put_bytes(self, key: str, body: bytes, content_type: str) -> None:
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType=content_type,
        )

    def put_file(self, key: str, path: Path, content_type: str) -> None:
        self.s3.upload_file(
            Filename=str(path),
            Bucket=self.bucket,
            Key=key,
            ExtraArgs={"ContentType": content_type},
        )

    def get_json(self, key: str) -> dict[str, Any] | None:
        try:
            out = self.s3.get_object(Bucket=self.bucket, Key=key)
        except self.s3.exceptions.NoSuchKey:
            return None
        return json.loads(out["Body"].read())
