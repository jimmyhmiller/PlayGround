
wrap : (f : a -> b) -> (comb: (a -> b) -> a -> b -> b) -> a -> b -> b
wrap f comb x = comb f x

c : (a -> b -> b) -> (a -> a) -> a -> b -> b
c f g x y = f (g x) y



add : (getSum : Int -> Int) -> (x :Int) -> (i : Int) -> Int
add getSum x i = getSum(i) + x


cadd : Int -> Int -> Int
cadd x y = x + y





wrap' : (comb : (a -> b) -> a -> b -> b) -> (f : a -> b) -> a -> b -> b
wrap' comb f x = comb f x



