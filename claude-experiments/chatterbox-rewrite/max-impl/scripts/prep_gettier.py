"""Extract Gettier paper body, clean PDF artifacts, split into chunks suitable
for TTS (chatterbox max_new_tokens = 1000 ~= 40s speech)."""
import re, sys, subprocess

raw = subprocess.run(
    ["pdftotext", "/home/jimmyhmiller/audiobooks/parked/Gettier.pdf", "-"],
    check=True, capture_output=True, text=True,
).stdout

# Trim to the body (starts at "IS JUSTIFIED TRUE BELIEF KNOWLEDGE?" heading).
m = re.search(r"IS JUSTIFIED TRUE BELIEF KNOWLEDGE\?", raw)
body = raw[m.end():]

# Strip footnote ref markers like "1, "2, etc. at line starts.
body = re.sub(r"^\d+\s+", "", body, flags=re.MULTILINE)

# Cut the trailing references (everything after the last paragraph before footnotes block).
# Heuristic: cut at first standalone-line-of-author-citation pattern.
cut = re.search(r"\n\s*1?\s*Plato seems to be considering", body)
if cut:
    body = body[:cut.start()]

# Drop page-number lines (just digits).
body = "\n".join(line for line in body.splitlines() if not re.fullmatch(r"\s*\d{1,3}\s*", line))

# Join wrapped lines into paragraphs (lines separated by single \n → space; \n\n → paragraph).
paragraphs = []
for para in re.split(r"\n\s*\n", body):
    para = re.sub(r"\s+", " ", para).strip()
    if len(para) > 5:
        paragraphs.append(para)

# Drop the "v ARIOUS" PDF dropcap artifact (paragraph starts with "v ARIOUS" - normalize to "Various").
paragraphs = [re.sub(r"^v\s+ARIOUS", "Various", p) for p in paragraphs]
# Other ALL-CAPS dropcap patterns (e.g. "L ET us", "S MITH").
paragraphs = [re.sub(r"^([A-Z])\s+([A-Z]+)", lambda m: m.group(1) + m.group(2).lower(), p) for p in paragraphs]

# Now split each paragraph into sentences (~150 chars each, but on sentence boundaries).
def sentences(text):
    # Split on sentence-ending punctuation followed by space + capital letter.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text)
    return [s.strip() for s in parts if s.strip()]

chunks = []
for p in paragraphs:
    for s in sentences(p):
        if len(s) < 5:
            continue
        chunks.append(s)

# Print summary, save chunks.
total_chars = sum(len(c) for c in chunks)
total_words = sum(len(c.split()) for c in chunks)
print(f"# chunks: {len(chunks)}", file=sys.stderr)
print(f"# total chars: {total_chars}, words: {total_words}", file=sys.stderr)
print(f"# avg chunk: {total_chars // max(len(chunks),1)} chars", file=sys.stderr)
print(f"# longest chunk: {max(len(c) for c in chunks)} chars", file=sys.stderr)

# Write JSON to stdout.
import json
print(json.dumps(chunks, indent=2))
