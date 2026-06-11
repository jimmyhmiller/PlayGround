xs = range(1, 10001)
print(sum(v for v in (x * x for x in xs) if v % 2 == 0))
