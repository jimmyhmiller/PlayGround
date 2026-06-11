def run():
    total = 0
    for i in range(2000):
        for j in range(2000):
            total += (i * j) % 7
    return total
print(run())
