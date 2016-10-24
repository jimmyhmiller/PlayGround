import System.Random
import Data.List
import Data.Ord

 
data Point = Point Float Float deriving (Show)
data Category = Red | Blue deriving (Show, Ord, Eq)
data Observation = Observation Point Category deriving (Show)


main = do
    g <- newStdGen
    let d = (generateData 1000 (0,20) pickCategory g)
    let reds = filter (checkColor (const True) Red) d
    let v = map (\n -> knearest 5 n d) d
    let cat = map classify v
    let incorrect = filter (checkColor (pointXPredicate (> 10)) Red) cat
    putStr $ show $ length $ incorrect

checkColor :: (Observation -> Bool) -> Category -> Observation -> Bool
checkColor pred color ob@(Observation (Point x _) obColor) = if ((pred ob) && (color /= obColor)) then True else False

pointXPredicate :: (Float -> Bool) -> Observation -> Bool
pointXPredicate pred (Observation (Point x _) _) = pred x

mostCommon :: Ord a => [a] -> a
mostCommon = head . maximumBy (comparing length) . group . sort
 
obsToCat :: Observation -> Category
obsToCat (Observation _ c) = c
 
classify :: [Observation] -> Observation
classify ((Observation p c):xs) = Observation p (mostCommon $ map obsToCat xs)
 
knearest :: Int -> Observation -> [Observation] -> [Observation]
knearest k o d = take k $ nearest o d
 
nearest :: Observation -> [Observation] -> [Observation]
nearest o d = sortBy (comparing $ distance o) d
 
distance :: Observation -> Observation -> Float
distance (Observation (Point x y) _) (Observation (Point x' y') _) = sqrt((x'-x)^2 + (y'-y)^2)
 
pairs :: [a] -> [(a,a)]
pairs [] = []
pairs (a:b:xs) = (a,b):pairs xs
 
pairToPoint :: (Float, Float) -> Point
pairToPoint (x,y) = Point x y
 
obsToPair :: Observation -> (Int, Int)
obsToPair (Observation (Point x y) _) = (floor x, floor y)
 
pickCategory :: Point -> Observation
pickCategory (Point x y)
    | x > 10 = Observation (Point x y) Red
    | otherwise = Observation (Point x y) Blue
 
generateData :: Int -> (Float, Float) -> (Point -> Observation) -> StdGen -> [Observation]
generateData n r f g = map (f . pairToPoint) rs
    where rs = pairs $ take (n * 2) $ randomRs r g