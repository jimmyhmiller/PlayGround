def run():
    acc = []
    for x in range(1, 10001):
        acc = acc + [x * x]          # immutable append, like funct's push
    out = []
    for v in acc:
        if v % 2 == 0:
            out = out + [v]
    total = 0
    for v in out:
        total += v
    return total
print(run())
