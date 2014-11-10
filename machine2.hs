data Auto a = Success [(a, Auto a)] | Start [(a, Auto a)] | State [(a, Auto a)] | End deriving (Eq, Show)

changeState :: Eq a => a -> Auto a ->  Auto a
changeState _ End = End
changeState s (Start []) = End
changeState s (State []) = End
changeState s (Success []) = End
changeState s (Start ((t, state):ts))
    | t == s = state
    | otherwise = changeState s (Start ts)
changeState s (Success ((t, state):ts))
    | t == s = state
    | otherwise = changeState s (Success ts)
changeState s (State ((t, state):ts))
    | t == s = state
    | otherwise = changeState s (State ts)

runMachine :: Eq a => [a] -> Auto a -> Bool
runMachine _ End = False
runMachine [] s = isSuccess s
runMachine (x:xs) s = runMachine xs $ changeState x s

s1 = Start [(0, s3)]
s2 = Success [(1, s1)]
s3 = State [(0, s3), (1, s2)]

isSuccess :: Auto a -> Bool
isSuccess (Success _) = True
isSuccess _ = False

main = putStrLn $ show $ runMachine [0, 1, 1, 0, 1] s1