"""Dump Mojo's bad-seed (chunk 53) tokens to a file for upstream to consume."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ['CHATTERBOX_BF16'] = '1'
os.environ['CHATTERBOX_CFM_STEPS'] = '5'
os.environ['CHATTERBOX_T3_FUSE_QKV'] = '1'
os.environ['CHATTERBOX_T3_FUSE_MLP'] = '1'

import numpy as np
from chatterbox_mojo import ChatterboxTTS
import op_t3, op_text_tokenize
from chatterbox_mojo.tts import punc_norm

REF = '/home/jimmyhmiller/.config/paper-audiobooks/default-voice.wav'
TEXT = ("What then could have motivated Carnap's heroic efforts on the conceptual "
        "side of epistemology, when hope of certainty on the doctrinal side was "
        "abandoned? There were two good reasons still.")
mojo = ChatterboxTTS.from_pretrained(use_bf16=True)
mojo.prepare_conditionals(REF, exaggeration=0.5)
text_n = punc_norm(TEXT)
text_ids = op_text_tokenize.tokenize(mojo._tok_h, text_n)
text_ids_full = [255] + list(text_ids) + [0]
raw = op_t3.generate(mojo._t3_h, mojo.conds.speaker_emb_256, mojo.conds.cond_prompt_tok,
                     text_ids_full, {'emotion': 0.5, 'cfg_weight': 0.5, 'temperature': 0.8,
                     'top_p': 0.95, 'rep_penalty': 1.2, 'min_p': 0.05, 'max_new': 1000,
                     'rng_seed': 0xDEADBEEF + 53})
EOS = 6562
mojo_tokens = [int(t) for t in raw if t != EOS and t < 6561]
print(f"mojo bad-seed tokens: {len(mojo_tokens)}")
np.save('/tmp/cfm_diag/mojo_bad_tokens.npy', np.array(mojo_tokens, dtype=np.int64))
print("saved /tmp/cfm_diag/mojo_bad_tokens.npy")
