import Test.QuickCheck

data Auto a = Accept [(a, Auto a)] | Start [(a, Auto a)] | StartAccept [(a, Auto a)] | State [(a, Auto a)] | End deriving (Eq, Show)

data Binary = Zero | One deriving (Eq, Show)

instance Arbitrary Binary where
    arbitrary = elements [Zero, One]

changeState :: Eq a => a -> Auto a ->  Auto a
changeState _ End = End
changeState _ (Start []) = End
changeState _ (State []) = End
changeState _ (Accept []) = End
changeState _ (StartAccept []) = End
changeState s (StartAccept ((t, state):ts))
    | t == s = state
    | otherwise = changeState s (StartAccept ts)
changeState s (Start ((t, state):ts))
    | t == s = state
    | otherwise = changeState s (Start ts)
changeState s (Accept ((t, state):ts))
    | t == s = state
    | otherwise = changeState s (Accept ts)
changeState s (State ((t, state):ts))
    | t == s = state
    | otherwise = changeState s (State ts)

runMachine :: Eq a => [a] -> Auto a -> Bool
runMachine _ End = False
runMachine [] s = isSuccess s
runMachine (x:xs) s = runMachine xs $ changeState x s

isSuccess :: Auto a -> Bool
isSuccess (Accept _) = True
isSuccess (StartAccept _) = True
isSuccess _ = False



s1 = StartAccept [(One, s2), (Zero, s5)]
s2 = Accept [(One, s2), (Zero, s3)]
s3 = State [(Zero, s3), (One, s4)]
s4 = Accept [(One, s4), (Zero, s3)]
s5 = Accept [(Zero, s5), (One, s6)]
s6 = State [(One, s6), (Zero, s7)]
s7 = Accept [(Zero, s7), (One, s6)]



correctPattern :: [Binary] -> Bool
correctPattern (One : One : xs) = correctPattern (One : xs)
correctPattern (Zero : Zero : xs) = correctPattern (Zero : xs)
correctPattern (Zero : One : xs) = wrongPattern (One : xs)
correctPattern (One : Zero : xs) = wrongPattern (Zero : xs)
correctPattern _ = True


wrongPattern :: [Binary] -> Bool
wrongPattern (One : One : xs) = wrongPattern (One : xs)
wrongPattern (Zero : Zero : xs) = wrongPattern (Zero : xs)
wrongPattern (Zero : One : xs) = correctPattern (One : xs)
wrongPattern (One : Zero : xs) = correctPattern (Zero : xs)
wrongPattern _ = False




prop_pattern_match xs = correctPattern xs == runMachine xs s1



main = verboseCheckWith stdArgs { maxSuccess = 5000 } prop_pattern_match

