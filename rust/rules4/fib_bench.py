import time

def fib(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    return fib(n - 1) + fib(n - 2)

print("Benchmarking fib(30)...")
start = time.time()
result = fib(30)
elapsed = time.time() - start
print(f"fib(30) = {result}")
print(f"Time: {elapsed:.3f}s")
