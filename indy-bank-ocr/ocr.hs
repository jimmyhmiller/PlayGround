import Data.List
import qualified Data.Map as Map
import Data.Maybe (fromJust, isJust, catMaybes)

numbers :: String
numbers = "\
\ _     _  _     _  _  _  _  _ \n\
\| |  | _| _||_||_ |_   ||_||_|\n\
\|_|  ||_  _|  | _||_|  ||_| _|"

splitEvery :: Int -> [a] -> [[a]]
splitEvery _ [] = []
splitEvery n xs = first : splitEvery n rest
  where (first, rest) = splitAt n xs

splitNumbers :: String -> [String]
splitNumbers nums = map (intercalate "") $ transpose $ map (splitEvery 3) $ lines nums

data SegmentState = On | Off deriving (Show, Eq, Ord)
data SevenSegment = Digit (SegmentState, SegmentState, SegmentState, SegmentState, SegmentState, SegmentState, SegmentState) deriving (Show, Eq, Ord)

data AccountNumber
  = Ill [Maybe Integer]
  | Err [Integer]
  | Amb [Integer]
  | Valid [Integer] deriving (Show, Eq, Ord)

isValid :: AccountNumber -> Bool
isValid (Valid _) = True
isValid _ = False

segState :: Char -> SegmentState
segState ' ' = Off
segState _ = On

toSevenSegment :: String -> SevenSegment
toSevenSegment str = case map segState str of
  [_, tm, _, mr, mm, ml, bl, bm, br] -> Digit (tm, mr, mm, ml, bl, bm, br)
  _ -> error "Should do better"


validSegments :: [SevenSegment]
validSegments = map toSevenSegment $ splitNumbers numbers

segmentLookup :: Map.Map Integer SevenSegment
segmentLookup = Map.fromList $ zip  [0..10] validSegments

numberLookup :: Map.Map SevenSegment Integer
numberLookup = Map.fromList $ zip validSegments [0..10]

sevenToInt :: SevenSegment -> Maybe Integer
sevenToInt segment = Map.lookup segment numberLookup

segmentsToAccountNumber :: [SevenSegment] -> [Maybe Integer]
segmentsToAccountNumber segments = map (flip Map.lookup numberLookup) segments

getAccountNumber :: String -> [Maybe Integer]
getAccountNumber line = segmentsToAccountNumber $ map toSevenSegment $ splitNumbers line

extractLines :: String -> [String]
extractLines s = map (intercalate "\n") $ splitEvery 3 $ lines s

convertAccountNumber :: [Maybe Integer] -> AccountNumber
convertAccountNumber nums = case nums of
  x | all isJust x && checkSum (catMaybes x) -> Valid $ catMaybes x
  x | all isJust x -> Err $ catMaybes x
  x -> Ill x

numOrQuestion :: Maybe Integer -> String
numOrQuestion (Just x) = show x
numOrQuestion Nothing = "?"

showAccountNumber :: AccountNumber -> String
showAccountNumber (Valid x) = intercalate "" (map show x)
showAccountNumber (Err x) = intercalate "" (map show x) ++ " ERR"
showAccountNumber (Ill x) = intercalate "" (map numOrQuestion x) ++ " ILL"

checkSum :: [Integer] -> Bool
checkSum nums = mod (sum $ zipWith (*) (reverse nums) [1..10]) 11 == 0

hammingDistance :: SevenSegment -> SevenSegment -> Int
hammingDistance
  (Digit (tm1, mr1, mm1, ml1, bl1, bm1, br1))
  (Digit (tm2, mr2, mm2, ml2, bl2, bm2, br2)) =
    length $ filter not $ zipWith (==)
    [tm1, mr1, mm1, ml1, bl1, bm1, br1]
    [tm2, mr2, mm2, ml2, bl2, bm2, br2]

candidateDigits :: SevenSegment -> [SevenSegment]
candidateDigits digit = filter ((== 1) . hammingDistance digit) validSegments

findPotentialAlternativeSegments :: [SevenSegment] -> [[SevenSegment]]
findPotentialAlternativeSegments segments = map candidateDigits segments

scenario1 :: String -> [[Integer]]
scenario1 s = map (map fromJust . getAccountNumber) $ extractLines s

scenario2 :: String -> [Bool]
scenario2 s = map (checkSum . map fromJust . getAccountNumber) $ extractLines s

scenario3 :: String -> [String]
scenario3 s = map (showAccountNumber. convertAccountNumber . getAccountNumber) $ extractLines s

scenario1File :: String
scenario1File = "./scenario1.txt"

scenario3File :: String
scenario3File = "./scenario3.txt"

main :: IO ()
main = do
  s <- readFile scenario3File
  -- print s
  print $ map (map sevenToInt) $ findPotentialAlternativeSegments [(segmentLookup Map.! 0), (segmentLookup Map.! 0)]
