#!/usr/bin/env python3
"""Quick test for the evaluator"""

import socket
import subprocess
import time

class BencodeEncoder:
    @staticmethod
    def encode(obj):
        if isinstance(obj, int):
            return f"i{obj}e".encode('utf-8')
        elif isinstance(obj, str):
            data = obj.encode('utf-8')
            return f"{len(data)}:".encode('utf-8') + data
        elif isinstance(obj, dict):
            result = b'd'
            for key, value in obj.items():
                result += BencodeEncoder.encode(key)
                result += BencodeEncoder.encode(value)
            result += b'e'
            return result
        else:
            raise ValueError(f"Cannot encode type: {type(obj)}")

class BencodeDecoder:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def decode(self):
        if self.pos >= len(self.data):
            return None
        c = chr(self.data[self.pos])
        if c == 'i':
            return self._decode_int()
        elif c == 'l':
            return self._decode_list()
        elif c == 'd':
            return self._decode_dict()
        elif c.isdigit():
            return self._decode_string()
        else:
            raise ValueError(f"Unknown bencode type: {c}")

    def _decode_list(self):
        self.pos += 1
        result = []
        while chr(self.data[self.pos]) != 'e':
            result.append(self.decode())
        self.pos += 1
        return result

    def _decode_int(self):
        self.pos += 1
        end = self.data.index(b'e', self.pos)
        val = int(self.data[self.pos:end])
        self.pos = end + 1
        return val

    def _decode_string(self):
        colon = self.data.index(b':', self.pos)
        length = int(self.data[self.pos:colon])
        self.pos = colon + 1
        string = self.data[self.pos:self.pos + length]
        self.pos += length
        return string

    def _decode_dict(self):
        self.pos += 1
        result = {}
        while chr(self.data[self.pos]) != 'e':
            key = self._decode_string().decode('utf-8')
            value = self.decode()
            result[key] = value
        self.pos += 1
        return result

# Start server
print("Starting server...")
server = subprocess.Popen(['./nrepl-server', '7889'],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
time.sleep(1)

try:
    # Connect
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("127.0.0.1", 7889))
    print("Connected!")

    # Test (+ 2 2)
    print("\nTesting: (+ 2 2)")
    msg = {"op": "eval", "code": "(+ 2 2)", "id": "1"}
    s.sendall(BencodeEncoder.encode(msg))
    data = s.recv(65536)
    decoder = BencodeDecoder(data)
    response = decoder.decode()

    def bytes_to_str(obj):
        if isinstance(obj, bytes):
            return obj.decode('utf-8')
        elif isinstance(obj, dict):
            return {bytes_to_str(k): bytes_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [bytes_to_str(item) for item in obj]
        else:
            return obj

    response = bytes_to_str(response)
    print(f"Response: {response}")
    print(f"Value: {response.get('value')}")

    s.close()
    print("\n✓ Test completed!")

finally:
    server.terminate()
    server.wait()
