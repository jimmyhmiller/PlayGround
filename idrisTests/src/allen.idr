data Optional : Type -> Type where
  Nothing : Optional a
  Something : a -> Optional a
  

instance Functor Optional where
    map f Nothing = Nothing
    map f (Something x) = Something (f x)


instance Applicative Optional where
    pure x = Something x
    (<*>) Nothing Nothing = Nothing
    (<*>) Nothing (Something x) = Nothing
    (<*>) (Something x) Nothing = Nothing
    (<*>) (Something f) (Something x) = Something (f x)


instance Monad Optional where
    (>>=) Nothing f = Nothing
    (>>=) (Something x) f = f x



instance Alternative Optional where
    empty = Nothing
    (<|>) Nothing Nothing = Nothing
    (<|>) Nothing x = x
    (<|>) x Nothing = x
    (<|>) x y = x


x : Optional Int
x = do
  a <- Something 2 <|> Nothing
  b <- Something 3 <|> Something 5
  c <- Nothing <|> Something 2
  return (a + b + c)
