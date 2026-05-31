#!/usr/bin/env python3
"""Minimal stand-in for the AWS Lambda Runtime API, to exercise an ai-lang
custom-runtime worker (`ai-lang run main` where main = lambda_serve(...))
exactly as Lambda would drive it, without deploying.

Usage: lambda_runtime_api_mock.py <port> <event-json> <response-out-file>

Serves one invocation: GET /2018-06-01/runtime/invocation/next returns the
event plus a request id header; the worker POSTs its result to
.../invocation/<id>/response, which we capture to <response-out-file>.
The worker then loops back to GET /next, which we hold open (Lambda's
/next is a long poll) so the orchestrator can stop the worker.
"""
import http.server
import sys
import time

PORT = int(sys.argv[1])
EVENT = sys.argv[2].encode()
OUT = sys.argv[3]
state = {"served": False, "captured": False}


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.endswith("/invocation/next"):
            if state["served"]:
                # Second poll: hold it open like a real long poll.
                time.sleep(60)
                return
            state["served"] = True
            self.send_response(200)
            self.send_header("Lambda-Runtime-Aws-Request-Id", "local-req-1")
            self.send_header("Content-Length", str(len(EVENT)))
            self.end_headers()
            self.wfile.write(EVENT)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        with open(OUT, "wb") as f:
            f.write(body)
        state["captured"] = True
        self.send_response(202)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *a):
        pass


http.server.HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
