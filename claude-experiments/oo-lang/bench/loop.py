import time
t0 = time.time()
acc = 0
for i in range(50000000):
    acc = (acc + i*3 + 7) % 1000000
t1 = time.time()
print(f"loop = {acc}  [{(t1-t0)*1000:.1f} ms]")
