import Data.Decimal

fact :: Decimal -> Decimal
fact 0 = 1
fact n = fact(n - 1) * n

f x = 1 / fact(x)

(|>) :: a -> (a -> b) -> b
(|>) a f = f a


main = [0..] 
    |> map f
    |> take 100
    |> foldl (+) 0 
    |> show
    |> putStrLn
