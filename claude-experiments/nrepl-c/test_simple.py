#!/usr/bin/env python3
import socket

def bencode_encode(obj):
    if isinstance(obj, int):
        return f"i{obj}e".encode('utf-8')
    elif isinstance(obj, str):
        data = obj.encode('utf-8')
        return f"{len(data)}:".encode('utf-8') + data
    elif isinstance(obj, dict):
        result = b'd'
        for key, value in obj.items():
            result += bencode_encode(key)
            result += bencode_encode(value)
        result += b'e'
        return result
    else:
        raise ValueError(f"Cannot encode type: {type(obj)}")

# Test connection
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("127.0.0.1", 7889))
print("Connected!")

# Send describe message
msg = {"op": "describe", "id": "1"}
encoded = bencode_encode(msg)
print(f"Sending: {encoded}")
s.sendall(encoded)

# Receive response
print("Waiting for response...")
data = s.recv(65536)
print(f"Received {len(data)} bytes: {data}")

s.close()
