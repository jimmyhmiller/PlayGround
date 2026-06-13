def steps(n):
    x = n; c = 0
    while x > 1:
        x = x // 2 if x % 2 == 0 else 3 * x + 1
        c += 1
    return c
def run():
    return sum(steps(n) for n in range(1, 100001))
print(run())
