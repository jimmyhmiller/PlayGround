"""WAV load/save roundtrip parity test (pure Mojo)."""
from std.testing import TestSuite, assert_almost_equal, assert_equal
from fixture import load_wav, save_wav


def test_wav_roundtrip() raises:
    var ref_path = "/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav"
    var tmp_path = "tests/fixtures/real_wavs/_roundtrip.wav"

    # Load the reference voice as fp32 samples.
    var w = load_wav(ref_path)
    print("loaded WAV samples:", w.numel(), " (=", Float64(w.numel()) / 24000.0, "s @ 24kHz)")
    assert_equal(w.rank(), 1)

    # Save it back and reload — should round-trip ~bit-exact.
    save_wav(tmp_path, w.data, 24000)
    var w2 = load_wav(tmp_path)
    assert_equal(w.numel(), w2.numel())

    var max_abs: Float32 = 0.0
    for i in range(w.numel()):
        var d = w.data[i] - w2.data[i]
        if d < 0.0: d = -d
        if d > max_abs: max_abs = d
        # int16 round-trip introduces ~1/32768 quantization error.
        assert_almost_equal(w.data[i], w2.data[i], atol=2.0 / 32768.0)
    print("max abs roundtrip diff:", max_abs)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
