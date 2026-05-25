"""Try external_call to dlsym-resolve libsoxr symbols."""
from std.sys import external_call
from std.testing import TestSuite


def test_soxr_smoke() raises:
    # soxr_version() returns const char* (a static string).
    var ver = external_call[
        "soxr_version",
        UnsafePointer[UInt8, MutAnyOrigin],
    ]()
    print("[soxr] called soxr_version")
    if Int(ver) == 0:
        print("[soxr] returned null")
    else:
        var s = String()
        var i = 0
        while i < 64:
            var ch = ver[i]
            if Int(ch) == 0: break
            s += chr(Int(ch))
            i += 1
        print("[soxr] version =", s)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
