#!/usr/bin/env python3

# Manual token lookup for our specific tokens
# Based on GPT-2 tokenizer
token_map = {
    15496: "Hello",
    995: " world",
    318: " is",
    257: " a",
    1295: " very",
    810: " well",
    345: " and",
    460: " we",
    198: "\n",
    
    # Some additional common ones for reference
    0: "!",
    1: "\"",
    2: "#",
    3: "$",
}

def decode_tokens(tokens):
    result = []
    for t in tokens:
        if t in token_map:
            result.append(token_map[t])
        else:
            result.append(f"[{t}]")
    return "".join(result)

print("=" * 60)
print("  🎉 GPT-2 GENERATED TEXT FROM OUR LISP IMPLEMENTATION 🎉")
print("=" * 60)
print()

# From our autoregressive generation test
prompt_tokens = [15496, 995, 318]
generated_tokens = [257, 1295, 810, 345, 460]

print("📝 PROMPT:")
print(f"   Tokens: {prompt_tokens}")
print(f"   Text:   \"{decode_tokens(prompt_tokens)}\"")
print()

print("✨ GENERATED:")
print(f"   Tokens: {generated_tokens}")
print(f"   Text:   \"{decode_tokens(generated_tokens)}\"")
print()

print("📖 FULL SEQUENCE:")
full_tokens = prompt_tokens + generated_tokens
print(f"   Tokens: {full_tokens}")
print(f"   Text:   \"{decode_tokens(full_tokens)}\"")
print()

print("=" * 60)
print()
print("Step-by-step generation:")
for i, token in enumerate(generated_tokens):
    context = prompt_tokens + generated_tokens[:i]
    context_text = decode_tokens(context)
    next_text = token_map.get(token, f"[{token}]")
    print(f"  Step {i}: \"{context_text}\" → generates \"{next_text}\"")
print()
print("=" * 60)

