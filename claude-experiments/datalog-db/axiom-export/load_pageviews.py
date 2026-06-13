#!/usr/bin/env python3
"""Load events.ndjson into datalog-db as PageView entities.

Speaks the datalog-db wire protocol directly (handshake + length-prefixed JSON
frames), batching asserts into transactions for throughput. Field names are
flattened (request.path -> path) and _time is parsed into an i64 epoch-ms `ts`
plus an i64 epoch-day `day` so per-day grouping needs no engine support.
"""
import json
import os
import socket
import struct
import sys
from datetime import datetime, timezone

HOST, PORT = "127.0.0.1", 5557
# VERSION 2 added a token auth frame to the handshake. The token comes from
# DATALOG_AUTH_TOKEN; an empty token is sent when the var is unset (only works
# against a server started with --no-auth).
MAGIC, VERSION = 0xDA7A1061, 2
AUTH_TOKEN = os.environ.get("DATALOG_AUTH_TOKEN", "").encode()
NDJSON = "events.ndjson"
BATCH = 500

# Axiom field -> PageView field. Anything not listed is dropped.
FIELD_MAP = {
    "request.host": "host",
    "request.path": "path",
    "vercel.route": "route",
    "request.method": "method",
    "request.ip": "ip",
    "request.referer": "referer",
    "request.userAgent": "userAgent",
    "vercel.region": "region",
    "vercel.source": "source",
    "request.vercelCache": "cache",
    "vercel.projectName": "project",
    "level": "level",
    "request.id": "requestId",
}
INT_MAP = {"request.statusCode": "status", "report.maxMemoryUsedMb": "memoryMb"}
FLOAT_MAP = {"report.durationMs": "durationMs"}


def parse_ts(s):
    # _time like 2026-05-11T14:33:28.067Z -> epoch ms (int).
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue
    return None


def to_op(ev):
    data = {}
    t = ev.get("_time")
    ms = parse_ts(t) if t else None
    if ms is None:
        return None  # an event with no usable time is not a page view we can index
    data["ts"] = ms
    data["day"] = ms // 86_400_000
    for src, dst in FIELD_MAP.items():
        v = ev.get(src)
        if v is not None and v != "":
            data[dst] = v
    for src, dst in INT_MAP.items():
        v = ev.get(src)
        if v is not None:
            data[dst] = int(v)
    for src, dst in FLOAT_MAP.items():
        v = ev.get(src)
        if v is not None:
            data[dst] = float(v)
    return {"assert": "PageView", "data": data}


class Conn:
    def __init__(self):
        self.s = socket.create_connection((HOST, PORT))
        # magic + version, then the auth frame: 4-byte length + token bytes.
        self.s.sendall(struct.pack(">II", MAGIC, VERSION))
        self.s.sendall(struct.pack(">I", len(AUTH_TOKEN)) + AUTH_TOKEN)
        # server replies with a single status byte: 0x00 = OK, 0x01 = error
        ack = self._recv_exact(1)
        if ack[0] != 0x00:
            ln = struct.unpack(">I", self._recv_exact(4))[0]
            raise IOError("handshake rejected: " + self._recv_exact(ln).decode())
        self.rid = 0

    def _recv_exact(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.s.recv(n - len(buf))
            if not chunk:
                raise IOError("connection closed")
            buf += chunk
        return buf

    def send(self, payload):
        self.rid += 1
        body = json.dumps(payload).encode()
        self.s.sendall(struct.pack(">QI", self.rid, len(body)) + body)
        hdr = self._recv_exact(12)
        _rid, length = struct.unpack(">QI", hdr)
        resp = self._recv_exact(length)
        return json.loads(resp)


def main():
    conn = Conn()
    ops, total, asserted = [], 0, 0
    with open(NDJSON) as f:
        for line in f:
            total += 1
            op = to_op(json.loads(line))
            if op:
                ops.append(op)
            if len(ops) >= BATCH:
                r = conn.send({"type": "transact", "ops": ops})
                if r.get("status") != "ok":
                    sys.exit(f"transact failed: {r}")
                asserted += len(ops)
                ops = []
                if asserted % 10000 == 0:
                    print(f"  asserted {asserted}", file=sys.stderr)
    if ops:
        r = conn.send({"type": "transact", "ops": ops})
        if r.get("status") != "ok":
            sys.exit(f"transact failed: {r}")
        asserted += len(ops)
    print(f"DONE: read {total} events, asserted {asserted} PageView entities",
          file=sys.stderr)


if __name__ == "__main__":
    main()
