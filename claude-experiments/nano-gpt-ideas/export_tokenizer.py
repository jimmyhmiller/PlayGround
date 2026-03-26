"""Export GPT-2 tokenizer for Rust usage."""
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.save_pretrained("gpt2_weights/tokenizer")
print("Saved tokenizer to gpt2_weights/tokenizer/")
