{-# LANGUAGE GeneralizedNewtypeDeriving #-}
import Test.QuickCheck

data Favorite = Icecream | CheeseCake | Nachos deriving (Eq, Show)

instance Arbitrary Favorite where
    arbitrary = elements [Icecream, CheeseCake, Nachos]

newtype IntegerLess = IntegerLess Integer deriving (Eq, Show, Arbitrary, Ord, Num, Enum, Real, Integral)
newtype IntegerGreater = IntegerGreater Integer deriving (Eq, Show, Arbitrary, Ord, Num, Enum, Real, Integral)
newtype IntegerEvenOdd = IntegerEvenOdd Integer deriving (Eq, Show, Arbitrary, Ord, Num, Enum, Real, Integral)

class Eq a => Poset a where
  (==<) :: a -> a -> Bool

instance Poset IntegerLess where
    a ==< b = a <= b

instance Poset IntegerGreater where
    a ==< b = a >= b

instance Poset Favorite where
    CheeseCake ==< Icecream = True
    x ==< y = x == y

instance Poset IntegerEvenOdd where
    x ==< y
        | even x && odd y = True
        | odd x && even y = False
        | otherwise = x <= y 


reflexive :: Poset a => a -> Bool
reflexive x = x ==< x == True

antisymetric :: Poset a => a -> a -> Bool
antisymetric x y = if x ==< y &&  y ==< x then x == y else True

transitive :: Poset a => a -> a -> a -> Bool
transitive x y z = if x ==< y && y ==< z then x ==< z else True


checkPosetIntLess :: IO ()
checkPosetIntLess = do    
    quickCheck (reflexive :: IntegerLess -> Bool)
    quickCheck (antisymetric :: IntegerLess -> IntegerLess -> Bool)
    quickCheck (transitive :: IntegerLess -> IntegerLess -> IntegerLess -> Bool)

checkPosetIntGreater :: IO ()
checkPosetIntGreater = do
    quickCheck (reflexive :: IntegerGreater -> Bool)
    quickCheck (antisymetric :: IntegerGreater -> IntegerGreater -> Bool)
    quickCheck (transitive :: IntegerGreater -> IntegerGreater -> IntegerGreater -> Bool)

checkPosetFavorite :: IO ()
checkPosetFavorite = do
    quickCheck (reflexive :: Favorite -> Bool)
    quickCheck (antisymetric :: Favorite -> Favorite -> Bool)
    quickCheck (transitive :: Favorite -> Favorite -> Favorite -> Bool)

checkPosetEvenOdd :: IO ()
checkPosetEvenOdd = do
    quickCheck (reflexive :: IntegerEvenOdd -> Bool)
    quickCheck (antisymetric :: IntegerEvenOdd -> IntegerEvenOdd -> Bool)
    quickCheck (transitive :: IntegerEvenOdd -> IntegerEvenOdd -> IntegerEvenOdd -> Bool)


main = do 
    putStrLn "Less"
    checkPosetIntLess
    putStrLn "Greater"
    checkPosetIntGreater
    putStrLn "Favorite"
    checkPosetFavorite
    putStrLn "Even Odd"
    checkPosetEvenOdd
