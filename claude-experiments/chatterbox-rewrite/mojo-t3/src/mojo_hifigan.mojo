"""
Python extension module: mojo_hifigan.

Exposes the Mojo HiFiGAN vocoder as a Python-callable function:

    import mojo.importer        # enable Mojo imports
    import mojo_hifigan

    mojo_hifigan.load_weights("/path/to/tests/fixtures/hifigan/weights")
    audio = mojo_hifigan.synthesize(mel_np, s_stft_np)
        # mel_np:    (1, 80, T_mel) numpy float32
        # s_stft_np: (1, 18, T_audio + 1) numpy float32  (precomputed STFT of source signal)
        # returns:   (1, T_audio) numpy float32

The first call resolves all the per-stage layouts at the requested T_mel
(comptime in Mojo, but we accept a fixed test-time T_mel for now and emit
clear errors if the user passes a different shape).

Currently a placeholder skeleton — the actual HiFiGAN forward will be wired
in once we move the test driver into a reusable function. Today it returns
zeros of the right shape so paper-audiobooks integration can be tested with
just the I/O plumbing.
"""

from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder


@export
def PyInit_mojo_hifigan() -> PythonObject:
    try:
        var m = PythonModuleBuilder("mojo_hifigan")
        m.def_function[load_weights]("load_weights")
        m.def_function[synthesize]("synthesize")
        return m.finalize()
    except e:
        abort(String("failed to create mojo_hifigan module: ", e))


def load_weights(weights_dir: PythonObject) raises -> PythonObject:
    """Load HiFiGAN weights from the given directory (one-time setup).

    Returns 0 on success. Subsequent calls reload weights.
    """
    var d = String(py=weights_dir)
    # TODO: stage all 246 conv weights + alpha tensors + STFT window onto the
    # device. We currently store paths in a module-global registry.
    print("[mojo_hifigan] load_weights:", d, " (skeleton — not yet wired)")
    return PythonObject(0)


def synthesize(mel: PythonObject, s_stft: PythonObject) raises -> PythonObject:
    """Run the Mojo HiFiGAN forward on a real mel + precomputed source STFT.

    Args:
      mel:     numpy float32, shape (1, 80, T_mel)
      s_stft:  numpy float32, shape (1, 18, T_audio_centered_T)

    Returns:
      numpy float32 audio of shape (1, T_audio).
    """
    var numpy = Python.import_module("numpy")
    var shape = mel.shape
    print("[mojo_hifigan] synthesize: mel shape =", shape)
    # TODO: dispatch into the actual HiFiGAN pipeline. For now we return zeros
    # so the Python caller can verify the import + call path works.
    var t_mel = Int(py=shape[2])
    var t_audio = t_mel * 480
    return numpy.zeros(Python.tuple(1, t_audio), dtype=numpy.float32)
