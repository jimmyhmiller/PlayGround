#!/usr/bin/env python3
"""Load the ENTIRE corpus (all 2,525 records under output/corpus/**, including
the nested bulk-S3 records) into datalog-db.

Unlike datalog_ingest.py (which was driven by the 26-doc index.jsonl), this is
driven by the records themselves — each record is self-contained with stem,
metadata, and full chapter bodies. Documents upsert on unique `stem`; chapters
upsert on composite `(doc, idx)`, so re-runs are idempotent.
"""
from __future__ import annotations

import json
import socket
import struct
import sys
import time
from pathlib import Path

HOST, PORT = "127.0.0.1", 5557
MAGIC, VERSION, MAX_PAYLOAD = 0xDA7A_1061, 1, 64 * 1024 * 1024
CORPUS = Path("output/corpus")

DOC_SCALAR = ["stem", "title", "source_name", "description", "content_sha256",
              "publish_year", "n_chapters", "total_chars", "title_source"]
DOC_LIST = ["authors", "subjects", "isbns", "dois"]


class Conn:
    def __init__(self, host=HOST, port=PORT):
        self.sock = socket.create_connection((host, port))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.sendall(struct.pack(">II", MAGIC, VERSION))
        ack = self._recvn(1)
        if ack[0] != 0x00:
            ln = struct.unpack(">I", self._recvn(4))[0]
            raise ConnectionError("handshake: " + self._recvn(ln).decode("utf-8", "replace"))
        self.rid = 0

    def _recvn(self, n):
        buf = bytearray()
        while len(buf) < n:
            c = self.sock.recv(n - len(buf))
            if not c:
                raise ConnectionError("server closed")
            buf += c
        return bytes(buf)

    def call(self, payload):
        body = json.dumps(payload).encode()
        if len(body) > MAX_PAYLOAD:
            return {"status": "error", "error": f"payload {len(body)} > MAX"}
        self.rid += 1
        self.sock.sendall(struct.pack(">QI", self.rid, len(body)) + body)
        self._recvn(8)
        plen = struct.unpack(">I", self._recvn(4))[0]
        return json.loads(self._recvn(plen))


def doc_data(rec: dict) -> dict:
    out = {}
    for f in DOC_SCALAR:
        v = rec.get(f)
        if v is None or (isinstance(v, str) and v == ""):
            continue
        if f == "publish_year":
            try:
                v = int(v)
            except (TypeError, ValueError):
                continue
        out[f] = v
    # local_* backfill when the enriched fields are empty
    if not out.get("title") and rec.get("local_title"):
        out["title"] = rec["local_title"]
    for f in DOC_LIST:
        items = [str(x) for x in (rec.get(f) or []) if x is not None and str(x) != ""]
        if not items and f == "authors" and rec.get("local_author"):
            items = [str(rec["local_author"])]
        if items:
            out[f] = items
    return out


def main():
    records = sorted(p for p in CORPUS.rglob("*.json"))
    print(f"{len(records)} json files under output/corpus/**")

    conn = Conn()
    print("connected:", conn.call({"type": "status"}))

    t0 = time.time()
    n_docs = n_ch = n_skip = 0
    biggest = 0
    # Build stem->id as we go so chapters can reference their doc in the same pass.
    for i, path in enumerate(records):
        try:
            rec = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(rec, dict) or "chapters" not in rec:
            n_skip += 1
            continue
        data = doc_data(rec)
        if "stem" not in data:
            n_skip += 1
            continue

        # Upsert the Document (unique stem). Returns its entity id.
        r = conn.call({"type": "transact",
                       "ops": [{"assert": "Document", "data": data}]})
        if r.get("status") != "ok":
            print(f"  doc err {data.get('stem')!r}: {r.get('error')}", file=sys.stderr)
            continue
        doc_id = r["data"]["entity_ids"][0]
        n_docs += 1

        # Chapters in one transaction (composite-key upsert on (doc, idx)).
        ops = []
        for idx, ch in enumerate(rec.get("chapters", [])):
            ops.append({"assert": "Chapter", "data": {
                "doc": {"ref": doc_id},
                "idx": idx,
                "title": ch.get("title") or "",
                "body": ch.get("body") or "",
                "n_chars": ch.get("n_chars") or len(ch.get("body") or ""),
            }})
        if ops:
            body_len = len(json.dumps({"type": "transact", "ops": ops}))
            if body_len > MAX_PAYLOAD - 4096:
                # Split oversized transactions (rare; very long books).
                for j in range(0, len(ops), 50):
                    conn.call({"type": "transact", "ops": ops[j:j + 50]})
            else:
                biggest = max(biggest, body_len)
                rr = conn.call({"type": "transact", "ops": ops})
                if rr.get("status") != "ok":
                    print(f"  ch err {data['stem']!r}: {rr.get('error')}", file=sys.stderr)
                    continue
            n_ch += len(ops)

        if (i + 1) % 200 == 0:
            dt = time.time() - t0
            print(f"  {i+1}/{len(records)}  docs={n_docs} chapters={n_ch}  "
                  f"({n_docs/dt:.0f} docs/s)", flush=True)

    dt = time.time() - t0
    print(f"\nDONE: {n_docs} documents, {n_ch} chapters in {dt:.1f}s "
          f"({n_skip} skipped); biggest tx {biggest/1e6:.1f}MB")


if __name__ == "__main__":
    main()
