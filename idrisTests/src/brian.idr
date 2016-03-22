%hide Either
%hide fact

data Color = Red | Green | Blue


isFavorite : Color -> Bool
isFavorite Red = False
isFavorite Green = True
isFavorite Blue = False

data Optional a = Nothing | Just a


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
    (<|>) Nothing x = x
    (<|>) x Nothing = x
    (<|>) x y = x

x : Optional Int
x = do
  a <- Just 3 <|> Nothing
  b <- Just 2 <|> Just 4
  c <- Nothing <|> Just 2
  pure (a + b + c)
