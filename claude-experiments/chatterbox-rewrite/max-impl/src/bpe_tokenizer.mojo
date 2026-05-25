"""
Pure-Mojo BPE tokenizer for the chatterbox text tokenizer.

Tokenizer characteristics (from tokenizer.json):
  - vocab size: 704
  - 265 merge rules in priority order
  - pre-tokenizer: Whitespace (split on whitespace)
  - special tokens: [STOP] (id 0), [UNK] (1), [SPACE] (2), [PAD], [SEP], [CLS], [MASK]
  - The chatterbox EnTokenizer replaces ' ' with '[SPACE]' BEFORE encoding.

Vocab + merges loaded from tab-separated text files at runtime.
"""
from std.collections import Dict
from std.pathlib import Path
from std.io.file import open


@fieldwise_init
struct Tokenizer(Movable):
    """BPE tokenizer state: vocab (token string → id) and merges (sorted by priority)."""
    var vocab: Dict[String, Int]
    var merges: List[List[String]]   # merges[i] = [a, b] with priority i (lower = higher priority)
    var merge_rank: Dict[String, Int]  # "a|b" -> priority


def load_tokenizer(vocab_path: String, merges_path: String) raises -> Tokenizer:
    """Load tab-separated vocab + merges files."""
    var vocab = Dict[String, Int]()
    var f = open(vocab_path, "r")
    var content = f.read()
    f.close()
    var line_start = 0
    var i = 0
    var n = content.byte_length()
    var bs = content.as_bytes()
    while i <= n:
        var at_end = i == n
        var is_nl = (not at_end) and Int(bs[i]) == 10
        if at_end or is_nl:
            if i > line_start:
                # Parse "token\tid".
                var line_len = i - line_start
                var tab_at = -1
                for j in range(line_len):
                    if Int(bs[line_start + j]) == 9:  # \t
                        tab_at = j
                        break
                if tab_at > 0:
                    var tok_bytes = List[UInt8]()
                    for j in range(tab_at):
                        tok_bytes.append(bs[line_start + j])
                    tok_bytes.append(0)  # null terminator for String construction
                    var tok = String("")
                    for j in range(tab_at):
                        tok += chr(Int(bs[line_start + j]))
                    var id_num = 0
                    for j in range(tab_at + 1, line_len):
                        var d = Int(bs[line_start + j]) - 48
                        if d < 0 or d > 9: continue
                        id_num = id_num * 10 + d
                    vocab[tok] = id_num
            line_start = i + 1
        i += 1

    var merges = List[List[String]]()
    var merge_rank = Dict[String, Int]()
    var f2 = open(merges_path, "r")
    var content2 = f2.read()
    f2.close()
    var line_start2 = 0
    var i2 = 0
    var n2 = content2.byte_length()
    var bs2 = content2.as_bytes()
    var rank = 0
    while i2 <= n2:
        var at_end = i2 == n2
        var is_nl = (not at_end) and Int(bs2[i2]) == 10
        if at_end or is_nl:
            if i2 > line_start2:
                var line_len = i2 - line_start2
                var tab_at = -1
                for j in range(line_len):
                    if Int(bs2[line_start2 + j]) == 9:
                        tab_at = j
                        break
                if tab_at > 0:
                    var a = String("")
                    for j in range(tab_at):
                        a += chr(Int(bs2[line_start2 + j]))
                    var b = String("")
                    for j in range(tab_at + 1, line_len):
                        b += chr(Int(bs2[line_start2 + j]))
                    var pair = List[String]()
                    pair.append(a.copy())
                    pair.append(b.copy())
                    merges.append(pair^)
                    merge_rank[a + "|" + b] = rank
                    rank += 1
            line_start2 = i2 + 1
        i2 += 1
    return Tokenizer(vocab^, merges^, merge_rank^)


def _split_to_chars(s: String) -> List[String]:
    """Split a UTF-8 string into single-char tokens. The chatterbox tokenizer's
    vocab uses single ASCII characters as initial tokens, so we split byte-wise."""
    var out = List[String]()
    var bs = s.as_bytes()
    var n = s.byte_length()
    var i = 0
    while i < n:
        var c = String("")
        c += chr(Int(bs[i]))
        out.append(c)
        i += 1
    return out^


def _apply_merges(mut tokens: List[String], tok: Tokenizer) raises:
    """Repeatedly apply the highest-priority adjacent pair merge until no more apply."""
    while True:
        var best_rank = -1
        var best_idx = -1
        for i in range(len(tokens) - 1):
            var key = tokens[i] + "|" + tokens[i + 1]
            if key in tok.merge_rank:
                var r = tok.merge_rank[key]
                if best_rank == -1 or r < best_rank:
                    best_rank = r
                    best_idx = i
        if best_idx < 0:
            break
        # Merge tokens[best_idx] and tokens[best_idx + 1].
        var lhs = tokens[best_idx].copy()
        var rhs = tokens[best_idx + 1].copy()
        var merged = lhs + rhs
        var new_tokens = List[String]()
        for i in range(len(tokens)):
            if i == best_idx:
                new_tokens.append(merged.copy())
            elif i == best_idx + 1:
                pass
            else:
                new_tokens.append(tokens[i].copy())
        tokens = new_tokens^


def _whitespace_split(s: String) -> List[String]:
    """Whitespace pre-tokenizer: split on whitespace, but preserve [SPACE] literal as its own token."""
    var out = List[String]()
    var current = String("")
    var bs = s.as_bytes()
    var n = s.byte_length()
    var i = 0
    while i < n:
        # Check if "[SPACE]" starts at i (7 chars).
        if i + 7 <= n:
            var matched = True
            var expected = String("[SPACE]")
            var eb = expected.as_bytes()
            for j in range(7):
                if Int(bs[i + j]) != Int(eb[j]):
                    matched = False
                    break
            if matched:
                if len(current) > 0:
                    out.append(current)
                    current = String("")
                out.append("[SPACE]")
                i += 7
                continue
        var ch = Int(bs[i])
        if ch == 32 or ch == 9 or ch == 10 or ch == 13:
            if len(current) > 0:
                out.append(current)
                current = String("")
        else:
            current += chr(ch)
        i += 1
    if len(current) > 0:
        out.append(current)
    return out^


def tokenize(text: String, tok: Tokenizer) raises -> List[Int]:
    """Full BPE tokenization: replace ' ' → '[SPACE]', whitespace-split, then BPE-merge each piece."""
    # 1. Replace ' ' with '[SPACE]' to match chatterbox EnTokenizer.encode().
    var replaced = String("")
    for c in text.codepoint_slices():
        var s = String(c)
        if s == " ":
            replaced += "[SPACE]"
        else:
            replaced += s

    # 2. Whitespace-pre-tokenize (split into words, keeping [SPACE] as own token).
    var pieces = _whitespace_split(replaced)

    # 3. For each piece: if it's a special token (like [SPACE]) look up directly.
    #    Otherwise: split to chars, apply merges, look up each.
    var ids = List[Int]()
    for piece in pieces:
        if piece in tok.vocab:
            # Direct lookup (handles [SPACE], [STOP], etc., and any single-char word in vocab).
            ids.append(tok.vocab[piece])
            continue
        var chars = _split_to_chars(piece)
        _apply_merges(chars, tok)
        for ch in chars:
            if ch in tok.vocab:
                ids.append(tok.vocab[ch])
            else:
                # UNK = id 1.
                ids.append(1)
    return ids^
