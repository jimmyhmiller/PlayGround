"""Compile-only check that all modules build clean."""
from std.testing import TestSuite

from modules import Linear, LayerNorm, RMSNorm, Embedding
from lstm import LSTMLayer
from voice_encoder import VoiceEncoder
from conv1d import Conv1d
from attention import qk_scaled_and_masked
from transformer_blocks import MHASelfAttention, LlamaMLP, MLP
from t3_block import T3Block
from s3tokenizer_block import FSMNAttention, S3TokenizerBlock
from t3 import T3
from s3tokenizer import S3Tokenizer
from mel_extractor import reflect_pad_1d


def test_compile_check() raises:
    print("All modules compiled")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
