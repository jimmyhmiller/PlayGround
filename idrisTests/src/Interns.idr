%hide Maybe


data Maybe a = Nothing | Just a


Functor Maybe where
  map func Nothing = Nothing
  map func (Just x) = Just (func x)

Applicative Maybe where
  pure x = Just x
  (<*>) Nothing Nothing = Nothing
  (<*>) Nothing (Just x) = Nothing
  (<*>) (Just x) Nothing = Nothing
  (<*>) (Just f) (Just x) = Just (f x)


Monad Maybe where
  (>>=) Nothing f = Nothing
  (>>=) (Just x) f = f x
  
Alternative Maybe where
  empty = Nothing
  (<|>) Nothing Nothing = Nothing
  (<|>) Nothing x = x
  (<|>) x y = x

thing : Maybe Int
thing = do 
  x <- Just 2
  y <- Nothing <|> Just 2
  z <- Just 2
  pure $ x + y + z

sumThem : List (Maybe Int) -> Int
sumThem [] = 0
sumThem (Nothing :: xs) = sumThem xs
sumThem ((Just x) :: xs) = x + sumThem xs

sumThem' : List Int -> List Int -> List Int
sumThem' xs ys = do 
  x <- xs
  y <- ys
  pure $ x + y




z : Maybe Int
z = map (+2) Nothing

--data List : Type -> Type where
--  Nil : List a
--  (::) : a -> List a -> List a


data Vector : Nat -> Type -> Type where
  Nil : Vector Z a
  (::) : a -> Vector n a -> Vector (S n) a
  

append : Vector n a -> Vector m a -> Vector (n + m) a
append [] y = y
append (x :: z) y = x :: append z y


q : Vector 2 Int  
q = 3 :: 2 :: Nil


x : Int
x = 2

double : Int -> Int
double x = x * 2


add : Int -> Int -> Int
add x y = x + y


data Color = Green | Blue | Red


myFavorite : Color -> Bool
myFavorite Green = True
myFavorite _ = False

