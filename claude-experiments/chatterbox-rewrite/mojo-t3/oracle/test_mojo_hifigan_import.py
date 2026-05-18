"""
Quick sanity test that the Mojo-built mojo_hifigan.so loads and can be called
from Python. Confirms the I/O plumbing works end-to-end.
"""
import sys
sys.path.insert(0, ".")
import numpy as np

import mojo_hifigan
print("imported:", mojo_hifigan.__name__)

rc = mojo_hifigan.load_weights("tests/fixtures/hifigan/weights")
print("load_weights rc:", rc)

mel = np.random.randn(1, 80, 32).astype(np.float32) * 0.5
s_stft = np.zeros((1, 18, 3841), dtype=np.float32)
audio = mojo_hifigan.synthesize(mel, s_stft)
print("audio shape:", audio.shape, "dtype:", audio.dtype, "peak:", np.abs(audio).max())
