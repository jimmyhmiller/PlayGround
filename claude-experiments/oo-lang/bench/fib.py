import time
def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)
t0 = time.time(); r = fib(32); t1 = time.time()
print(f"fib(32) = {r}  [{(t1-t0)*1000:.1f} ms]")
