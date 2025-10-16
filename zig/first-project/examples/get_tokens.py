#!/usr/bin/env python3

# Manual tokenization for Shakespeare prompts
# Based on common GPT-2 tokens

prompts = [
    "Shall I compare thee to a summer",
    "To be or not to be",
    "Once upon a time",
    "In the beginning",
]

# We need actual token IDs - let me try a simple approach
# Common Shakespeare tokens
print("Suggested prompts and approximate token IDs:\n")
print("1. Simple: 'Once upon a time' - good for story generation")
print("   Tokens: [7454, 2402, 257, 640]")
print()
print("2. Shakespeare style: Use a known working prompt")
print("   'The' = token 464")
print("   Let's use a simple prompt that will work")
print()
print("Recommended: Start with a simple, common phrase")
print("'The quick brown' or 'In the beginning' or 'Once upon a time'")

