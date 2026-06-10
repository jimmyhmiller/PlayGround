#!/usr/bin/env python3
"""Ingest the paper-audiobooks corpus into datalog-db as a stress test.

Talks the raw TCP protocol directly:
  handshake: magic 0xDA7A_1061 (u32 BE) + version 1 (u32 BE)  -> 8 bytes
  message:   request_id (u64 BE) | payload_len (u32 BE) | payload_json
  MAX_PAYLOAD_SIZE = 64 MiB

Stages:
  1. Ingest all 2,525 index.jsonl rows as Document entities.
  2. Ingest the 28 full-body records' chapters as Chapter entities (ref -> Document).

Prints timings and surfaces every server error verbatim.
"""
from __future__ import annotations

import json
import socket
import struct
import sys
import time
from pathlib import Path

HOST, PORT = "127.0.0.1", 5557
MAGIC = 0xDA7A_1061
VERSION = 1
MAX_PAYLOAD = 64 * 1024 * 1024

CORPUS = Path("output/corpus")
INDEX = CORPUS / "index.jsonl"


class Conn:
    def __init__(self, host=HOST, port=PORT):
        self.sock = socket.create_connection((host, port))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.sendall(struct.pack(">II", MAGIC, VERSION))
        self.rid = 0
        # Handshake ack: 0x00 = OK; 0x01 + u32 len + msg = error.
        ack = self._recvn(1)
        if ack[0] != 0x00:
            ln = struct.unpack(">I", self._recvn(4))[0]
            raise ConnectionError("handshake failed: " + self._recvn(ln).decode("utf-8", "replace"))

    def _recvn(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("server closed connection")
            buf += chunk
        return bytes(buf)

    def call(self, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        if len(body) > MAX_PAYLOAD:
            return {"status": "error", "error": f"payload {len(body)} > MAX_PAYLOAD {MAX_PAYLOAD}"}
        self.rid += 1
        self.sock.sendall(struct.pack(">QI", self.rid, len(body)) + body)
        _req_id = struct.unpack(">Q", self._recvn(8))[0]
        plen = struct.unpack(">I", self._recvn(4))[0]
        return json.loads(self._recvn(plen))

    def close(self):
        self.sock.close()


def clean(d: dict, fields: list[str]) -> dict:
    """Keep only declared fields with meaningful values.

    Drops None (db rejects null) and empty lists/strings to keep datoms lean —
    an absent optional field is the natural representation of "no data". Empty
    *lists* in particular would otherwise store a useless `[]` datom per doc.
    """
    out = {}
    for f in fields:
        v = d.get(f)
        if v is None:
            continue
        if isinstance(v, (list, str)) and len(v) == 0:
            continue
        out[f] = v
    return out


# Scalar + list fields present on every index.jsonl row. The list fields
# (authors/subjects/isbns/dois) map to the new `[string]` columns.
DOC_FIELDS = ["stem", "title", "source_name", "authors", "subjects",
              "isbns", "dois", "publish_year",
              "n_chapters", "total_chars", "title_source",
              "has_description", "enriched_source"]

# Extra fields only present in the full-body records (the 26 with bodies).
# Merged onto the Document when we ingest its chapters.
DOC_EXTRA_FIELDS = ["description", "content_sha256"]

# Fields the schema declares as `[string]` — every element must be a string.
LIST_STRING_FIELDS = ["authors", "subjects", "isbns", "dois"]


def coerce_doc_types(data: dict) -> None:
    """Make values match the declared column types in place.

    - List columns are `[string]`: stringify each element (ISBNs sometimes
      parse as ints) and drop empties.
    - `publish_year` is i64: coerce a stringy/float year to int, else drop it.
    """
    for f in LIST_STRING_FIELDS:
        if f in data:
            items = [str(x) for x in data[f] if x is not None and str(x) != ""]
            if items:
                data[f] = items
            else:
                del data[f]
    if "publish_year" in data:
        try:
            data["publish_year"] = int(data["publish_year"])
        except (TypeError, ValueError):
            del data["publish_year"]


def ingest_documents(conn: Conn) -> dict[str, int]:
    """Returns {stem: entity_id}."""
    rows = [json.loads(l) for l in INDEX.read_text().splitlines() if l.strip()]
    print(f"[docs] {len(rows)} index rows")

    stem_to_id: dict[str, int] = {}
    batch, batch_stems = [], []
    BATCH = 500
    t0 = time.time()
    n_ok = 0

    def flush():
        nonlocal n_ok
        if not batch:
            return
        resp = conn.call({"type": "transact", "ops": batch})
        if resp.get("status") != "ok":
            print(f"  ERROR on batch ({len(batch)} ops): {resp.get('error')}", file=sys.stderr)
            # try one-by-one to find offender
            for op, stem in zip(batch, batch_stems):
                r = conn.call({"type": "transact", "ops": [op]})
                if r.get("status") != "ok":
                    print(f"    bad doc stem={stem!r}: {r.get('error')}", file=sys.stderr)
                else:
                    stem_to_id[stem] = r["data"]["entity_ids"][0]
                    n_ok += 1
        else:
            ids = resp["data"]["entity_ids"]
            for stem, eid in zip(batch_stems, ids):
                stem_to_id[stem] = eid
            n_ok += len(ids)
        batch.clear()
        batch_stems.clear()

    for row in rows:
        data = clean(row, DOC_FIELDS)
        if "stem" not in data:
            continue
        coerce_doc_types(data)
        batch.append({"assert": "Document", "data": data})
        batch_stems.append(row["stem"])
        if len(batch) >= BATCH:
            flush()
    flush()

    dt = time.time() - t0
    print(f"[docs] {n_ok} inserted in {dt:.2f}s ({n_ok/dt:.0f}/s)")
    return stem_to_id


def ingest_chapters(conn: Conn, stem_to_id: dict[str, int]) -> None:
    full_records = sorted(p for p in CORPUS.glob("*.json"))
    print(f"[chapters] {len(full_records)} full-body records on disk")
    t0 = time.time()
    n_ch = 0
    n_docs = 0
    biggest_payload = 0
    for path in full_records:
        rec = json.loads(path.read_text())
        if not isinstance(rec, dict) or "chapters" not in rec:
            continue  # control files (s3_manifest.json, ocr_backlog.json) are lists
        stem = rec.get("stem")
        doc_id = stem_to_id.get(stem)
        if doc_id is None:
            print(f"  WARN: full record {stem!r} not in index; skipping", file=sys.stderr)
            continue
        ops = []
        # Patch the Document with full-record-only fields (description,
        # content_sha256), updating by entity id. local_title/local_author
        # backfill title/authors when the index lacked them.
        extra = clean(rec, DOC_EXTRA_FIELDS)
        if rec.get("local_author") and not rec.get("authors"):
            extra["authors"] = [str(rec["local_author"])]
        coerce_doc_types(extra)
        if extra:
            ops.append({"assert": "Document", "entity": doc_id, "data": extra})
        for i, ch in enumerate(rec.get("chapters", [])):
            data = {
                "doc": {"ref": doc_id},
                "idx": i,
                "title": ch.get("title") or "",
                "body": ch.get("body") or "",
                "n_chars": ch.get("n_chars") or len(ch.get("body") or ""),
            }
            ops.append({"assert": "Chapter", "data": data})
        # one transaction per doc (could be large — that's the point)
        body_len = len(json.dumps({"type": "transact", "ops": ops}))
        biggest_payload = max(biggest_payload, body_len)
        resp = conn.call({"type": "transact", "ops": ops})
        if resp.get("status") != "ok":
            print(f"  ERROR doc={stem!r} ({len(ops)} chapters, {body_len/1e6:.1f}MB): {resp.get('error')}", file=sys.stderr)
            continue
        n_ch += len(ops)
        n_docs += 1
    dt = time.time() - t0
    print(f"[chapters] {n_ch} chapters across {n_docs} docs in {dt:.2f}s; biggest tx payload={biggest_payload/1e6:.2f}MB")


def main():
    conn = Conn()
    print("connected.", conn.call({"type": "status"}))
    stem_to_id = ingest_documents(conn)
    ingest_chapters(conn, stem_to_id)
    conn.close()


if __name__ == "__main__":
    main()
