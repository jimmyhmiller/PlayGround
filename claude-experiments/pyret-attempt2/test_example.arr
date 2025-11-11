fun factorial(n):
  if n == 0:
    1
  else:
    n * factorial(n - 1)
  end
end

check:
  factorial(5) is 120
  factorial(0) is 1
end
