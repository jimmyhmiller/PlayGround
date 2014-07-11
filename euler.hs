import Data.Numbers.Primes


main = putStrLn $ show $ euler1

intSqrt n = floor $ sqrt $ fromIntegral n

(|>) x y = y x

fibs = 1 : 1 : zipWith (+) fibs (tail fibs)

is_prime n = case (n) of 
	 1 -> False
	 2 -> True
	 n -> not $ any (\x -> n `mod` x == 0) (takeWhile (<= (intSqrt n)) primes)


prime_factors n = [x | x <- (takeWhile (<= (intSqrt n)) primes), n `mod` x == 0]

euler1 = sum [x | x <- [0..999], x `mod` 3 == 0 || x `mod` 5 == 0]

euler2 = sum [ x | x <- takeWhile (<= 4000000) fibs, even x]

euler3 = maximum $ prime_factors 600851475143

euler4 = maximum $ [z | x <- [100..999], y <- [x..999], let z = x*y, show z == (reverse $ show z)]

euler5 = foldr1 lcm [1..20]

euler6 = (sum [1..100])^2 - (sum $ map (^2) [1..100])

euler7 = primes !! 10000