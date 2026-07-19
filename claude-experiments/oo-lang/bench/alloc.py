import time
class Point:
    __slots__ = ("x", "y")
    def __init__(self, x, y):
        self.x = x; self.y = y
t0 = time.time()
pts = []
for i in range(90000):
    pts.append(Point(i, i*2))
sum_ = 0
for p in pts:
    sum_ += p.x + p.y
t1 = time.time()
print(f"alloc sum = {sum_}  [{(t1-t0)*1000:.1f} ms]")
