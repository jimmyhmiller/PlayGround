

data Optional : Type -> Type where
  Nothing : Optional a
  Just : a -> Optional a



x : Optional Int
x = Nothing

y : Optional Int
y = Just 4


instance Functor Optional where
    map f Nothing = Nothing
    map f (Just x) = Just (f x)
    
    
instance Applicative Optional where
    pure x = Just x
    (<*>) Nothing y = Nothing
    (<*>) (Just f) Nothing = Nothing
    (<*>) (Just f) (Just y) = Just (f y)
    
    
instance Monad Optional where
    (>>=) Nothing f = Nothing
    (>>=) (Just x) f = f x
    

    
instance Alternative Optional where    
    empty = Nothing
    (<|>) Nothing Nothing = Nothing
    (<|>) Nothing y = y
    (<|>) x Nothing = x
    (<|>) x y = x

    
n : Optional Int
n = do
  a <- Just 3 <|> Nothing
  b <- Just 5
  c <- Nothing <|> Just 5
  return (a + b + c)

