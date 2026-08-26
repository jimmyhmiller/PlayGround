#!/usr/bin/env python3
"""lspcli — a good-faith ("steelman") LSP bridge for coding agents.

Wraps basedpyright-langserver behind a warm daemon + a stateless CLI:
    python3 lspcli.py <op> --root R [--file F --line N --col M] [...]

Ops: symbols, definition, references, hover, rename, diagnostics, shutdown.
Positions are 1-based lines and columns. The daemon owns the LSP session,
handshake, and document sync (files re-read from disk before every request,
so agent edits are always visible). First call auto-starts the daemon.
"""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time

SYMKIND = {1: "file", 2: "module", 3: "namespace", 4: "package", 5: "class",
           6: "method", 7: "property", 8: "field", 9: "constructor",
           12: "function", 13: "variable", 14: "constant"}
SEVERITY = {1: "error", 2: "warning", 3: "information", 4: "hint"}


def sock_path(root):
    h = hashlib.md5(os.path.abspath(root).encode()).hexdigest()[:10]
    return os.path.join("/tmp", f"lspd-{h}.sock")


def uri_of(root, rel):
    return "file://" + os.path.join(os.path.abspath(root), rel)


def rel_of(root, uri):
    path = uri[len("file://"):]
    return os.path.relpath(path, os.path.abspath(root))


# ---------------------------------------------------------------- daemon

class LspServer:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.proc = subprocess.Popen(
            ["basedpyright-langserver", "--stdio"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, cwd=self.root)
        self.next_id = 1
        self.pending = {}          # id -> threading.Event + slot
        self.diags = {}            # uri -> published diagnostics
        self.docs = {}             # uri -> version
        self.lock = threading.Lock()
        threading.Thread(target=self._reader, daemon=True).start()
        self._initialize()

    # -- framing
    def _send(self, msg):
        data = json.dumps(msg).encode()
        head = f"Content-Length: {len(data)}\r\n\r\n".encode()
        with self.lock:
            self.proc.stdin.write(head + data)
            self.proc.stdin.flush()

    def _read_msg(self):
        length = None
        while True:
            line = self.proc.stdout.readline()
            if not line:
                return None
            line = line.strip()
            if not line:
                break
            if line.lower().startswith(b"content-length:"):
                length = int(line.split(b":")[1])
        return json.loads(self.proc.stdout.read(length))

    def _reader(self):
        while True:
            msg = self._read_msg()
            if msg is None:
                return
            if "id" in msg and "method" in msg:        # server -> client request
                self._answer_server(msg)
            elif "method" in msg:                       # notification
                if msg["method"] == "textDocument/publishDiagnostics":
                    p = msg["params"]
                    self.diags[p["uri"]] = p["diagnostics"]
            elif "id" in msg:                           # response
                slot = self.pending.pop(msg["id"], None)
                if slot:
                    slot["msg"] = msg
                    slot["ev"].set()

    def _answer_server(self, msg):
        method, mid = msg["method"], msg["id"]
        if method == "workspace/configuration":
            result = [{} for _ in msg["params"]["items"]]
        elif method == "workspace/workspaceFolders":
            result = [{"uri": "file://" + self.root, "name": "ws"}]
        else:                                           # registerCapability etc.
            result = None
        self._send({"jsonrpc": "2.0", "id": mid, "result": result})

    def request(self, method, params, timeout=45):
        with self.lock:
            rid = self.next_id
            self.next_id += 1
        slot = {"ev": threading.Event(), "msg": None}
        self.pending[rid] = slot
        self._send({"jsonrpc": "2.0", "id": rid, "method": method,
                    "params": params})
        if not slot["ev"].wait(timeout):
            raise RuntimeError(f"LSP request {method} timed out after {timeout}s")
        msg = slot["msg"]
        if "error" in msg:
            raise RuntimeError(f"LSP error for {method}: {msg['error']}")
        return msg.get("result")

    def notify(self, method, params):
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _initialize(self):
        self.request("initialize", {
            "processId": os.getpid(),
            "rootUri": "file://" + self.root,
            "workspaceFolders": [{"uri": "file://" + self.root, "name": "ws"}],
            "capabilities": {
                "textDocument": {
                    "publishDiagnostics": {},
                    "diagnostic": {"dynamicRegistration": False},
                    "hover": {"contentFormat": ["plaintext", "markdown"]},
                },
                "workspace": {"configuration": True,
                              "workspaceFolders": True,
                              "symbol": {}},
            },
        }, timeout=90)
        self.notify("initialized", {})

    # -- document sync: disk is the truth, always
    def sync(self, rel):
        full = os.path.join(self.root, rel)
        with open(full, encoding="utf-8") as f:
            text = f.read()
        uri = uri_of(self.root, rel)
        if uri not in self.docs:
            self.docs[uri] = 1
            self.notify("textDocument/didOpen", {"textDocument": {
                "uri": uri, "languageId": "python", "version": 1,
                "text": text}})
        else:
            self.docs[uri] += 1
            self.notify("textDocument/didChange", {
                "textDocument": {"uri": uri, "version": self.docs[uri]},
                "contentChanges": [{"text": text}]})
        return uri

    # -- op helpers
    def loc_out(self, loc):
        uri = loc.get("uri") or loc.get("targetUri")
        rng = loc.get("range") or loc.get("targetSelectionRange")
        rel = rel_of(self.root, uri)
        line = rng["start"]["line"] + 1
        text = ""
        try:
            with open(os.path.join(self.root, rel), encoding="utf-8") as f:
                lines = f.read().splitlines()
            if 1 <= line <= len(lines):
                text = lines[line - 1].rstrip()
        except OSError:
            pass
        return {"file": rel, "line": line,
                "col": rng["start"]["character"] + 1, "text": text}

    def resolve_symbol(self, name):
        res = self.request("workspace/symbol", {"query": name}) or []
        exact = [s for s in res if s["name"] == name]
        return exact if exact else res

    def pos_params(self, rel, line, col):
        uri = self.sync(rel)
        return {"textDocument": {"uri": uri},
                "position": {"line": line - 1, "character": col - 1}}


def handle(server, req):
    op = req["op"]
    root = server.root
    if op == "ping":
        return {"ok": True}

    # allow --symbol NAME addressing on position ops (workspace/symbol)
    if req.get("symbol") and op in ("definition", "references", "hover",
                                    "rename"):
        cands = server.resolve_symbol(req["symbol"])
        if not cands:
            return {"ok": False, "error": f"no workspace symbol matches "
                                          f"{req['symbol']!r}"}
        if len(cands) > 1 and req.get("index") is None:
            return {"ok": False, "error": "ambiguous symbol; pass file/line/"
                                          "col or index",
                    "candidates": [dict(server.loc_out(c["location"]),
                                        name=c["name"],
                                        kind=SYMKIND.get(c["kind"],
                                                         c["kind"]))
                                   for c in cands]}
        pick = cands[req.get("index") or 0]
        loc = server.loc_out(pick["location"])
        req = dict(req, file=loc["file"], line=loc["line"], col=loc["col"])
        # workspace/symbol points at the name; nudge col to land on it
        line_text = loc["text"]
        name = req["symbol"].split(".")[-1]
        idx = line_text.find(name)
        if idx >= 0:
            req["col"] = idx + 1

    if op == "symbols":
        res = server.request("workspace/symbol", {"query": req["query"]})
        return {"ok": True, "symbols": [
            dict(server.loc_out(s["location"]), name=s["name"],
                 kind=SYMKIND.get(s["kind"], s["kind"]),
                 container=s.get("containerName"))
            for s in (res or [])]}

    if op == "diagnostics":
        rels = [req["file"]] if req.get("file") else sorted(
            os.path.relpath(os.path.join(dp, f), root)
            for dp, dns, fns in os.walk(root)
            if not any(part.startswith(".") for part in dp.split(os.sep))
            for f in fns if f.endswith(".py"))
        out = []
        for rel in rels:
            uri = server.sync(rel)
            try:
                res = server.request("textDocument/diagnostic",
                                     {"textDocument": {"uri": uri}})
                items = res.get("items", [])
            except RuntimeError:
                time.sleep(1.0)
                items = server.diags.get(uri, [])
            for d in items:
                out.append({"file": rel,
                            "line": d["range"]["start"]["line"] + 1,
                            "severity": SEVERITY.get(d.get("severity"), "?"),
                            "message": d["message"],
                            "rule": d.get("code")})
        errors = [d for d in out if d["severity"] == "error"]
        return {"ok": True, "diagnostics": out, "errors": len(errors),
                "files_checked": len(rels)}

    params = server.pos_params(req["file"], req["line"], req["col"])

    if op == "definition":
        res = server.request("textDocument/definition", params) or []
        if isinstance(res, dict):
            res = [res]
        return {"ok": True, "definitions": [server.loc_out(x) for x in res]}

    if op == "references":
        params["context"] = {"includeDeclaration":
                             bool(req.get("include_declaration", True))}
        res = server.request("textDocument/references", params) or []
        return {"ok": True, "references": [server.loc_out(x) for x in res],
                "count": len(res)}

    if op == "hover":
        res = server.request("textDocument/hover", params)
        contents = ""
        if res and res.get("contents"):
            c = res["contents"]
            contents = c.get("value") if isinstance(c, dict) else str(c)
        return {"ok": True, "hover": contents}

    if op == "rename":
        params["newName"] = req["new_name"]
        res = server.request("textDocument/rename", params)
        if not res:
            return {"ok": False, "error": "rename returned no edit"}
        changes = {}
        if "changes" in res:
            for uri, edits in res["changes"].items():
                changes[rel_of(root, uri)] = edits
        for dc in res.get("documentChanges", []):
            if "textDocument" in dc:
                changes[rel_of(root, dc["textDocument"]["uri"])] = dc["edits"]
        summary = {rel: len(edits) for rel, edits in changes.items()}
        if not req.get("apply"):
            detail = {rel: [{"line": e["range"]["start"]["line"] + 1,
                             "newText": e["newText"]} for e in edits]
                      for rel, edits in changes.items()}
            return {"ok": True, "applied": False, "edits": summary,
                    "detail": detail}
        for rel, edits in changes.items():
            full = os.path.join(root, rel)
            with open(full, encoding="utf-8") as f:
                lines = f.read().split("\n")
            for e in sorted(edits, key=lambda e: (e["range"]["start"]["line"],
                                                  e["range"]["start"]["character"]),
                            reverse=True):
                sl, sc = e["range"]["start"]["line"], e["range"]["start"]["character"]
                el, ec = e["range"]["end"]["line"], e["range"]["end"]["character"]
                if sl == el:
                    lines[sl] = lines[sl][:sc] + e["newText"] + lines[sl][ec:]
                else:
                    lines[sl:el + 1] = [lines[sl][:sc] + e["newText"]
                                        + lines[el][ec:]]
            with open(full, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            server.sync(rel)
        return {"ok": True, "applied": True, "edits": summary}

    return {"ok": False, "error": f"unknown op {op!r}"}


def run_daemon(root):
    path = sock_path(root)
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    server = LspServer(root)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(8)
    srv.settimeout(60)
    last = time.time()
    while True:
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            if time.time() - last > 1200:       # idle exit
                break
            continue
        last = time.time()
        try:
            data = b""
            while not data.endswith(b"\n"):
                chunk = conn.recv(65536)
                if not chunk:
                    break
                data += chunk
            req = json.loads(data)
            if req.get("op") == "shutdown":
                conn.sendall(b'{"ok": true, "shutdown": true}\n')
                conn.close()
                break
            try:
                resp = handle(server, req)
            except Exception as e:
                resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            conn.sendall(json.dumps(resp).encode() + b"\n")
            conn.close()
        except Exception:
            pass
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def client_call(root, req, autostart=True):
    path = sock_path(root)
    for attempt in range(2):
        try:
            c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            c.settimeout(90)
            c.connect(path)
            c.sendall(json.dumps(req).encode() + b"\n")
            data = b""
            while not data.endswith(b"\n"):
                chunk = c.recv(65536)
                if not chunk:
                    break
                data += chunk
            return json.loads(data)
        except (FileNotFoundError, ConnectionRefusedError):
            if not autostart or attempt:
                raise
            log = open(os.path.join("/tmp", "lspd.log"), "ab")
            subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), "daemon",
                 "--root", root],
                stdout=log, stderr=log, start_new_session=True)
            for _ in range(120):
                if os.path.exists(path):
                    time.sleep(0.3)
                    break
                time.sleep(0.25)
    raise RuntimeError("could not reach daemon")


def main():
    ap = argparse.ArgumentParser(prog="lspcli")
    ap.add_argument("op", help="symbols|definition|references|hover|rename|"
                               "diagnostics|shutdown|daemon|ping")
    ap.add_argument("--root", default=".")
    ap.add_argument("--file"); ap.add_argument("--line", type=int)
    ap.add_argument("--col", type=int); ap.add_argument("--symbol")
    ap.add_argument("--index", type=int)
    ap.add_argument("--query"); ap.add_argument("--new-name",
                                               dest="new_name")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--no-declaration", action="store_true")
    args = ap.parse_args()
    if args.op == "daemon":
        run_daemon(args.root)
        return 0
    req = {"op": args.op}
    for k in ("file", "line", "col", "symbol", "index", "query", "new_name"):
        v = getattr(args, k)
        if v is not None:
            req[k] = v
    if args.apply:
        req["apply"] = True
    if args.no_declaration:
        req["include_declaration"] = False
    resp = client_call(args.root, req)
    print(json.dumps(resp, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
