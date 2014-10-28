import Data.Map as Map
import Control.Applicative
import Control.Monad
import Data.Set as Set

data DFA a b = DFA (Map a (Map b a)) (Set a)  deriving (Eq, Show)


auto = recurFromList [("q1", 
                        [(0, "q1"), 
                         (1, "q2")]),
                      ("q2",
                        [(1, "q2"),
                         (0, "q3")]),
                      ("q3",
                        [(0, "q2"),
                         (1, "q2")])]

                      

machine = DFA auto (Set.fromList ["q2"])


recurFromList :: (Ord a, Ord b) => [(a, [(b, a)])] -> Map a (Map b a)
recurFromList [(s,kv)] = Map.fromList $ [(s, Map.fromList kv)]
recurFromList (x:xs) = Map.union (recurFromList [x]) (recurFromList xs)


transition (DFA m _) i s =  (join $ liftM2 Map.lookup s (return m)) >>= Map.lookup i


automate (DFA _ a) [] s = liftM2 Set.member s (return a)
automate m (i:is) s = automate m is $ transition m i s


main = putStrLn $ show $ automate machine [0, 0, 1, 1, 1] (Just "q1")