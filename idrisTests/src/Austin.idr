

factorial : (Int -> Int) -> Int -> Int
factorial f 0 = 1
factorial f n = n * f (n - 1)


fix : (a -> a) -> a 
fix f = let x = 2 * x in x



