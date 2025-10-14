#!/bin/bash

echo "Starting server..."
./nrepl-server 7889 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

sleep 1

echo -e "\nSending message..."
python3 test_simple.py

echo -e "\nKilling server..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
