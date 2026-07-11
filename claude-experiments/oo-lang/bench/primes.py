import time
def is_prime(n):
    if n < 2: return False
    d = 2
    while d*d <= n:
        if n % d == 0: return False
        d += 1
    return True
t0 = time.time()
count = sum(1 for n in range(2, 200000) if is_prime(n))
t1 = time.time()
print(f"primes<200000 = {count}  [{(t1-t0)*1000:.1f} ms]")
