"""Dump upstream Chatterbox EnTokenizer output for a few test strings."""
import os, sys


def main():
    from chatterbox.models.tokenizers.tokenizer import EnTokenizer

    # The EnTokenizer ctor takes a path to its merged spm/vocab file.
    # Find it in the chatterbox checkpoint directory.
    CKPT = "/home/jimmyhmiller/.cache/huggingface/hub/models--ResembleAI--chatterbox/snapshots/ef85ce7bef2f3f1a74d0d837d379d2fcb68203cd"
    # Tokenizer file is typically named "tokenizer.json" or similar.
    import os
    candidates = [
        os.path.join(CKPT, "tokenizer.json"),
        os.path.join(CKPT, "tokenizers", "tokenizer.json"),
    ]
    tok_path = None
    for c in candidates:
        if os.path.exists(c):
            tok_path = c
            break
    if tok_path is None:
        print("Looking in", CKPT, ":")
        for f in os.listdir(CKPT):
            print("  ", f)
        return

    print("Using tokenizer:", tok_path)
    t = EnTokenizer(tok_path)

    for text in ["hello world", "the quick brown fox"]:
        ids = t.text_to_tokens(text).tolist()
        if isinstance(ids[0], list): ids = ids[0]
        print(f"  '{text}' → {ids}")


if __name__ == "__main__":
    main()
