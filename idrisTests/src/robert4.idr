%hide map

reduce : (a -> b -> b) -> b -> List a -> b
reduce f x [] = x
reduce f x (y :: xs) = f y (reduce f x xs)

fmap : (a -> b) -> List a -> List b
fmap f xs = reduce ((::) . f) [] xs

ffilter : (a -> Bool) -> List a -> List a
ffilter pred xs = reduce (\x, ys => if pred x then x :: ys else ys) [] xs


mapif : (a -> Bool) -> (a -> a) -> List a -> List a
mapif pred f xs = reduce (\x, ys => if pred x then f x :: ys else x :: ys) [] xs
