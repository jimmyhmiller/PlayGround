#!/usr/bin/env python3
"""
Comprehensive test suite for nREPL server implementation.
Tests compatibility with standard nREPL clients.
"""

import socket
import subprocess
import time
import sys
import signal
from typing import Optional, Dict, Any, List

class BencodeDecoder:
    """Simple bencode decoder for parsing nREPL responses."""

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def decode(self) -> Any:
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

    def _decode_int(self) -> int:
        self.pos += 1  # skip 'i'
        end = self.data.index(b'e', self.pos)
        val = int(self.data[self.pos:end])
        self.pos = end + 1
        return val

    def _decode_string(self) -> bytes:
        colon = self.data.index(b':', self.pos)
        length = int(self.data[self.pos:colon])
        self.pos = colon + 1
        string = self.data[self.pos:self.pos + length]
        self.pos += length
        return string

    def _decode_list(self) -> list:
        self.pos += 1  # skip 'l'
        result = []
        while chr(self.data[self.pos]) != 'e':
            result.append(self.decode())
        self.pos += 1  # skip 'e'
        return result

    def _decode_dict(self) -> dict:
        self.pos += 1  # skip 'd'
        result = {}
        while chr(self.data[self.pos]) != 'e':
            key = self._decode_string().decode('utf-8')
            value = self.decode()
            result[key] = value
        self.pos += 1  # skip 'e'
        return result

class BencodeEncoder:
    """Simple bencode encoder for creating nREPL messages."""

    @staticmethod
    def encode(obj: Any) -> bytes:
        if isinstance(obj, int):
            return f"i{obj}e".encode('utf-8')
        elif isinstance(obj, bytes):
            return f"{len(obj)}:".encode('utf-8') + obj
        elif isinstance(obj, str):
            data = obj.encode('utf-8')
            return f"{len(data)}:".encode('utf-8') + data
        elif isinstance(obj, list):
            return b'l' + b''.join(BencodeEncoder.encode(item) for item in obj) + b'e'
        elif isinstance(obj, dict):
            result = b'd'
            for key, value in obj.items():
                result += BencodeEncoder.encode(key)
                result += BencodeEncoder.encode(value)
            result += b'e'
            return result
        else:
            raise ValueError(f"Cannot encode type: {type(obj)}")

class NReplClient:
    """Simple nREPL client for testing."""

    def __init__(self, host: str = "127.0.0.1", port: int = 7888):
        self.host = host
        self.port = port
        self.socket: Optional[socket.socket] = None
        self.msg_id = 0

    def connect(self) -> bool:
        """Connect to nREPL server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            self.socket.connect((self.host, self.port))
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False

    def disconnect(self):
        """Disconnect from nREPL server."""
        if self.socket:
            self.socket.close()
            self.socket = None

    def send_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message and receive response."""
        if not self.socket:
            raise RuntimeError("Not connected")

        # Add message ID if not present
        if "id" not in msg:
            self.msg_id += 1
            msg["id"] = str(self.msg_id)

        # Encode and send
        encoded = BencodeEncoder.encode(msg)
        self.socket.sendall(encoded)

        # Receive response
        data = self.socket.recv(65536)
        if not data:
            raise RuntimeError("Connection closed")

        # Decode response
        decoder = BencodeDecoder(data)
        response = decoder.decode()

        # Convert bytes to strings for easier handling
        return self._bytes_to_str(response)

    def _bytes_to_str(self, obj: Any) -> Any:
        """Recursively convert bytes to strings in response."""
        if isinstance(obj, bytes):
            return obj.decode('utf-8')
        elif isinstance(obj, dict):
            return {self._bytes_to_str(k): self._bytes_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._bytes_to_str(item) for item in obj]
        else:
            return obj

class NReplTestSuite:
    """Test suite for nREPL server."""

    def __init__(self, port: int = 7888):
        self.port = port
        self.server_process: Optional[subprocess.Popen] = None
        self.passed = 0
        self.failed = 0
        self.tests_run = 0

    def start_server(self) -> bool:
        """Start the nREPL server."""
        print(f"Starting nREPL server on port {self.port}...")
        try:
            self.server_process = subprocess.Popen(
                ['./nrepl-server', str(self.port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd='/home/jimmyhmiller/Documents/Code/Playground/claude-experiments/nrepl-c'
            )
            time.sleep(1)  # Give server time to start

            # Check if server is still running
            if self.server_process.poll() is not None:
                print("Server failed to start")
                return False

            print("Server started successfully")
            return True
        except Exception as e:
            print(f"Failed to start server: {e}")
            return False

    def stop_server(self):
        """Stop the nREPL server."""
        if self.server_process:
            print("\nStopping server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            self.server_process = None

    def test(self, name: str, func):
        """Run a test and track results."""
        self.tests_run += 1
        print(f"\nTest {self.tests_run}: {name}")
        try:
            func()
            self.passed += 1
            print(f"  ✓ PASSED")
        except AssertionError as e:
            self.failed += 1
            print(f"  ✗ FAILED: {e}")
        except Exception as e:
            self.failed += 1
            print(f"  ✗ ERROR: {e}")

    def assert_equal(self, actual, expected, msg: str = ""):
        """Assert equality."""
        if actual != expected:
            raise AssertionError(f"{msg}\n  Expected: {expected}\n  Got: {actual}")

    def assert_in(self, item, container, msg: str = ""):
        """Assert item in container."""
        if item not in container:
            raise AssertionError(f"{msg}\n  {item} not in {container}")

    def assert_true(self, condition, msg: str = ""):
        """Assert condition is true."""
        if not condition:
            raise AssertionError(f"{msg}\n  Condition was false")

    def test_basic_connection(self):
        """Test that we can connect to the server."""
        client = NReplClient(port=self.port)
        self.assert_true(client.connect(), "Failed to connect to server")
        client.disconnect()

    def test_describe_op(self):
        """Test the describe operation."""
        client = NReplClient(port=self.port)
        client.connect()

        response = client.send_message({"op": "describe"})

        # Verify response structure
        self.assert_in("ops", response, "Response missing 'ops'")
        self.assert_in("versions", response, "Response missing 'versions'")
        self.assert_in("status", response, "Response missing 'status'")

        # Verify required ops
        ops = response["ops"]
        self.assert_in("describe", ops, "Missing 'describe' op")
        self.assert_in("clone", ops, "Missing 'clone' op")
        self.assert_in("close", ops, "Missing 'close' op")
        self.assert_in("eval", ops, "Missing 'eval' op")

        # Verify status
        self.assert_in("done", response["status"], "Status should contain 'done'")

        # Verify message ID is echoed
        self.assert_in("id", response, "Response missing message 'id'")

        client.disconnect()

    def test_clone_session(self):
        """Test session cloning."""
        client = NReplClient(port=self.port)
        client.connect()

        response = client.send_message({"op": "clone"})

        # Verify response
        self.assert_in("new-session", response, "Response missing 'new-session'")
        self.assert_in("status", response, "Response missing 'status'")
        self.assert_in("done", response["status"], "Status should contain 'done'")

        # Verify session ID format (should be a UUID)
        session_id = response["new-session"]
        self.assert_true(len(session_id) == 36, f"Session ID should be 36 chars (UUID), got {len(session_id)}")
        self.assert_true(session_id.count('-') == 4, "Session ID should have 4 hyphens (UUID format)")

        client.disconnect()

    def test_multiple_sessions(self):
        """Test creating multiple sessions."""
        client = NReplClient(port=self.port)
        client.connect()

        sessions = []
        for i in range(5):
            response = client.send_message({"op": "clone"})
            self.assert_in("new-session", response, f"Failed to create session {i+1}")
            sessions.append(response["new-session"])

        # Verify all sessions are unique
        self.assert_equal(len(sessions), len(set(sessions)), "Sessions should be unique")

        client.disconnect()

    def test_close_session(self):
        """Test closing a session."""
        client = NReplClient(port=self.port)
        client.connect()

        # Create a session
        clone_response = client.send_message({"op": "clone"})
        session_id = clone_response["new-session"]

        # Close the session
        close_response = client.send_message({
            "op": "close",
            "session": session_id
        })

        self.assert_in("status", close_response, "Response missing 'status'")
        self.assert_in("done", close_response["status"], "Status should contain 'done'")

        client.disconnect()

    def test_close_nonexistent_session(self):
        """Test closing a nonexistent session."""
        client = NReplClient(port=self.port)
        client.connect()

        response = client.send_message({
            "op": "close",
            "session": "00000000-0000-0000-0000-000000000000"
        })

        self.assert_in("status", response, "Response missing 'status'")
        self.assert_in("error", response["status"], "Status should contain 'error' for nonexistent session")

        client.disconnect()

    def test_eval_basic(self):
        """Test basic eval operation."""
        client = NReplClient(port=self.port)
        client.connect()

        # Create a session first
        clone_response = client.send_message({"op": "clone"})
        session_id = clone_response["new-session"]

        # Eval some code
        response = client.send_message({
            "op": "eval",
            "code": "(+ 1 2)",
            "session": session_id
        })

        self.assert_in("status", response, "Response missing 'status'")
        self.assert_in("done", response["status"], "Status should contain 'done'")
        self.assert_in("ns", response, "Response missing 'ns' (namespace)")

        # Should have a value (even if it's just nil for now)
        self.assert_in("value", response, "Response missing 'value'")

        client.disconnect()

    def test_eval_without_session(self):
        """Test eval without a session."""
        client = NReplClient(port=self.port)
        client.connect()

        response = client.send_message({
            "op": "eval",
            "code": "(+ 1 2)"
        })

        # Should still work (server should handle sessionless eval)
        self.assert_in("status", response, "Response missing 'status'")
        self.assert_in("done", response["status"], "Status should contain 'done'")

        client.disconnect()

    def test_message_id_echo(self):
        """Test that message IDs are echoed back."""
        client = NReplClient(port=self.port)
        client.connect()

        msg_id = "test-message-123"
        response = client.send_message({
            "op": "describe",
            "id": msg_id
        })

        self.assert_in("id", response, "Response missing 'id'")
        self.assert_equal(response["id"], msg_id, "Message ID not echoed correctly")

        client.disconnect()

    def test_session_echo(self):
        """Test that session IDs are echoed back."""
        client = NReplClient(port=self.port)
        client.connect()

        # Create a session
        clone_response = client.send_message({"op": "clone"})
        session_id = clone_response["new-session"]

        # Use the session
        response = client.send_message({
            "op": "describe",
            "session": session_id
        })

        self.assert_in("session", response, "Response missing 'session'")
        self.assert_equal(response["session"], session_id, "Session ID not echoed correctly")

        client.disconnect()

    def test_unknown_op(self):
        """Test handling of unknown operations."""
        client = NReplClient(port=self.port)
        client.connect()

        response = client.send_message({"op": "unknown-operation"})

        self.assert_in("status", response, "Response missing 'status'")
        self.assert_in("error", response["status"], "Status should contain 'error' for unknown op")

        client.disconnect()

    def test_multiple_clients(self):
        """Test that multiple clients can connect simultaneously."""
        client1 = NReplClient(port=self.port)
        client2 = NReplClient(port=self.port)

        # Connect both clients
        self.assert_true(client1.connect(), "Client 1 failed to connect")
        self.assert_true(client2.connect(), "Client 2 failed to connect")

        # Both should be able to communicate
        response1 = client1.send_message({"op": "describe"})
        response2 = client2.send_message({"op": "describe"})

        self.assert_in("ops", response1, "Client 1 got invalid response")
        self.assert_in("ops", response2, "Client 2 got invalid response")

        client1.disconnect()
        client2.disconnect()

    def test_session_isolation(self):
        """Test that sessions are isolated between clients."""
        client1 = NReplClient(port=self.port)
        client2 = NReplClient(port=self.port)

        client1.connect()
        client2.connect()

        # Create sessions
        session1 = client1.send_message({"op": "clone"})["new-session"]
        session2 = client2.send_message({"op": "clone"})["new-session"]

        # Sessions should be different
        self.assert_true(session1 != session2, "Sessions should be unique across clients")

        # Client 1 should be able to close its session
        response = client1.send_message({"op": "close", "session": session1})
        self.assert_in("done", response["status"], "Failed to close session")

        # Client 2's session should still be valid (test by using it)
        response = client2.send_message({
            "op": "describe",
            "session": session2
        })
        self.assert_in("ops", response, "Client 2's session should still be valid")

        client1.disconnect()
        client2.disconnect()

    def run_all_tests(self):
        """Run all tests."""
        print("="*60)
        print("nREPL Server Test Suite")
        print("="*60)

        if not self.start_server():
            print("Failed to start server. Exiting.")
            return False

        try:
            # Core protocol tests
            self.test("Basic connection", self.test_basic_connection)
            self.test("Describe operation", self.test_describe_op)
            self.test("Message ID echo", self.test_message_id_echo)
            self.test("Unknown operation handling", self.test_unknown_op)

            # Session tests
            self.test("Clone session", self.test_clone_session)
            self.test("Multiple sessions", self.test_multiple_sessions)
            self.test("Close session", self.test_close_session)
            self.test("Close nonexistent session", self.test_close_nonexistent_session)
            self.test("Session ID echo", self.test_session_echo)
            self.test("Session isolation", self.test_session_isolation)

            # Eval tests
            self.test("Basic eval", self.test_eval_basic)
            self.test("Eval without session", self.test_eval_without_session)

            # Multi-client tests
            self.test("Multiple clients", self.test_multiple_clients)

        finally:
            self.stop_server()

        # Print summary
        print("\n" + "="*60)
        print("Test Summary")
        print("="*60)
        print(f"Tests run: {self.tests_run}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")

        if self.failed == 0:
            print("\n✓ All tests passed!")
            return True
        else:
            print(f"\n✗ {self.failed} test(s) failed")
            return False

def main():
    """Main entry point."""
    port = 7888
    if len(sys.argv) > 1:
        port = int(sys.argv[1])

    suite = NReplTestSuite(port=port)
    success = suite.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
