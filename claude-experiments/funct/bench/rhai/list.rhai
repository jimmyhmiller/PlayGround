let xs = [];
for x in 1..=10000 { xs.push(x); }
let r = xs.map(|x| x * x).filter(|x| x % 2 == 0).reduce(|sum, x| sum + x, 0);
print(r);
