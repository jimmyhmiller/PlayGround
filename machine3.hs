data DFA st ab = DFA st (st -> ab -> st) (st -> Bool) 

data Binary = Zero | One deriving (Eq, Show)


sameOneZeroAndZeroOne :: DFA (Maybe Binary, Int, Int) Binary
sameOneZeroAndZeroOne = DFA (Nothing, 0, 0) delta sameNum
    where
        delta (Just One, n, m) Zero = (Just Zero, n + 1, m)
        delta (Just Zero, n, m) One = (Just One, n, m + 1)
        delta (b, n, m) b' = (Just b', n, m)
        sameNum (_, n, m) = n == m


sameOneZeroAndZeroOne' :: DFA (Maybe Binary, Maybe Binary) Binary
sameOneZeroAndZeroOne' = DFA (Nothing, Nothing) delta sameAsInit
    where
        delta (Nothing, Nothing) Zero = (Just Zero, Just Zero)
        delta (Nothing, Nothing) One = (Just One, Just One)
        delta (init, Just One) Zero = (init, Just Zero)
        delta (init, Just Zero) One = (init, Just One)
        delta (init, prev) _  = (init, prev)
        sameAsInit (init, end) = init == end



runDFA :: DFA a b -> [b] -> Bool
runDFA (DFA state transition final) [] = final state
runDFA (DFA state transition final) (x:xs) = runDFA (DFA (transition state x) transition final) xs


main :: IO ()
main = putStrLn $ show $ runDFA sameOneZeroAndZeroOne' [One, Zero, Zero, Zero, One]