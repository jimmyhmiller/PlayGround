from std.math import sqrt

def main():
    var total: Float64 = 0.0
    for i in range(1, 10001):
        total += sqrt(Float64(i))
    print("sum =", total)
