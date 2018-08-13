import Test.QuickCheck

data Point = Point Int Int deriving Show


isPos :: Int -> Bool
isPos = (> 0)

posPoint :: Point -> Point
posPoint (Point x y) = Point (abs x) (abs y)

pointIsPos :: Point -> Bool
pointIsPos p = case (posPoint p) of 
    (Point x y) -> isPos x && isPos y 

main :: IO ()
main = do
    quickCheck pointIsPos
