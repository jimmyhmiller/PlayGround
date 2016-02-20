

data Array : Nat -> Type -> Type where
  Empty : Array Z Int
  Append : Int -> Array n Int -> Array (S n) Int
  
  
total
factorial : Nat -> Nat
factorial Z = 1
factorial n@(S x) = n * (factorial x)


r: factorial(Z) = 1
r = Refl

r1: factorial(5) = 120
r1 = Refl



instance Num String where
    (+) x y = x ++ y
    (*) x y = "stuff"
    fromInteger x = cast x
    
    



first : Array (S n) Int -> Int
first (Append x y) = x


third : Array (S (S (S n))) Int -> Int
third (Append x y) = case y of
                          (Append x z) => (case z of
                                                (Append a b) => a)



data Optional : Type -> Type where
  Nothing : Optional a
  Just : a -> Optional a


instance Functor Optional where
    map f Nothing = Nothing
    map f (Just x) = Just (f x)


instance Applicative Optional where
    pure x = Just x
    (<*>) Nothing Nothing = Nothing
    (<*>) Nothing (Just x) = Nothing
    (<*>) (Just x) Nothing = Nothing
    (<*>) (Just f) (Just x) = Just (f x)
    
    
instance Monad Optional where
    (>>=) Nothing f = Nothing
    (>>=) (Just x) f = f x



instance Alternative Optional where
    empty = Nothing
    (<|>) Nothing Nothing = Nothing
    (<|>) Nothing (Just x) = Just x
    (<|>) (Just x) Nothing = Just x
    (<|>) (Just x) (Just y) = Just x


x : Optional Int
x = do
  a <- Just 3 <|> Nothing
  b <- Just 4 <|> Just 7
  c <- Nothing <|> Just 5
  return (a + b + c)
 

  
y : Optional String
y = do
  a <- Just "stuff" <|> Nothing
  b <- Just "stuff" <|> Just "stuff"
  c <- Nothing <|> Just "stuff"
  return (a ++ b ++ c)
