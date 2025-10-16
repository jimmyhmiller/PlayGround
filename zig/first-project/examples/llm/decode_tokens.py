tokens = [15496, 995, 318, 257, 1295, 810, 345, 460, 307, 3511, 13, 198, 198, 464, 1266, 835, 284, 651, 257, 1365]

# Approximate decoding (GPT-2 tokenizer mappings)
token_map = {
    15496: "Hello",
    995: " world",
    318: " is",
    257: " a",
    1295: " very",
    810: " well",
    345: " and",
    460: " we",
    307: " are",
    3511: " happy",
    13: ".",
    198: "\n",
    464: " The",
    1266: " best",
    835: " way",
    284: " to",
    651: " get",
    1365: " good"
}

text = "".join([token_map.get(t, f"[{t}]") for t in tokens])
print(f"Generated text:\n{text}")
