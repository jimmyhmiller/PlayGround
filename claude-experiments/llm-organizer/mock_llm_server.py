#!/usr/bin/env python3
"""
Mock LLM server for testing llm-organizer.
Responds to POST requests with simple test responses.
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json

class MockLLMHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        try:
            request = json.loads(post_data.decode('utf-8'))
            prompt = request.get('prompt', '')

            # Generate appropriate response based on prompt
            if 'Summarize' in prompt or 'summarize' in prompt:
                response_text = "This document discusses machine learning topics including neural networks and deep learning."
            elif 'tag' in prompt.lower() or 'keyword' in prompt.lower():
                response_text = '["machine-learning", "AI", "neural-networks", "deep-learning"]'
            elif 'categor' in prompt.lower():
                response_text = '["Technology/AI", "Research"]'
            elif 'entities' in prompt.lower() or 'Extract key entities' in prompt:
                response_text = '{"people": [], "organizations": [], "dates": [], "locations": []}'
            elif 'SQL WHERE' in prompt or 'sql' in prompt.lower():
                response_text = "file_type = 'text/plain'"
            else:
                response_text = "Test response from mock LLM server"

            response = {
                "text": response_text,
                "finish_reason": "stop"
            }

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))

            print(f"Request: {prompt[:100]}...")
            print(f"Response: {response_text[:100]}...")

        except Exception as e:
            print(f"Error: {e}")
            self.send_response(500)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress default logging
        pass

if __name__ == '__main__':
    server = HTTPServer(('localhost', 8081), MockLLMHandler)
    print("Mock LLM server running on http://localhost:8081")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
