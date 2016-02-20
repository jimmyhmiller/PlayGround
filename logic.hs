import Prelude hiding (and, or)
import Data.List (nub)
import Data.Tree
import Data.Tree.Pretty
import Data.Char (toLower)

data Operation = And | Or | If | Iff deriving (Show, Eq)

data Statement = T
               | F
               | Neg Statement
               | Prop String 
               | Op Operation Statement Statement deriving (Eq)


instance Show Statement where
    show (Prop p) = p
    show T = "T"
    show F = "F"
    show (Neg p) = "~" ++ show p
    show (Op If p q) = "(" ++ show p ++ " " ++ "->" ++ " " ++ show q ++ ")"
    show (Op Iff p q) = "(" ++ show p ++ " " ++ "<->" ++ " " ++ show q ++ ")"
    show (Op o p q) =  "(" ++ show p ++ " " ++ map toLower (show o) ++ " " ++ show q ++ ")"




or p q = Op Or p q
and p q = Op And p q



s = (Op If (Op Or (Op And (Prop "p") (Prop "q")) (Op And (Prop "p'") (Prop "q'"))) (Op Iff (Prop "q") (Prop "r")))
main = putStrLn $ drawVerticalTree $ fmap show $ fullLogicTree (Prop "p" `or` Neg (Prop "p"))

unique :: Eq a => [a] -> [a]
unique a = nub a

removeTAnd :: Statement -> Statement
removeTAnd (Op And T q) = q
removeTAnd (Op And p T) = p
removeTAnd s = s

removeFOr :: Statement -> Statement
removeFOr (Op Or F q) = q
removeFOr (Op Or p F) = p
removeFOr s = s

reduceFAnd :: Statement -> Statement
reduceFAnd (Op And F q) = F
reduceFAnd (Op And p F) = F
reduceFAnd s = s

reduceTOr :: Statement -> Statement
reduceTOr (Op Or T q) = T
reduceTOr (Op Or p T) = T
reduceTOr s = s

reduceTAntec :: Statement -> Statement
reduceTAntec (Op If T q) = q
reduceTAntec s = s

reduceFAntec :: Statement -> Statement
reduceFAntec (Op If F _) = T
reduceFAntec s = s

reduceTCons :: Statement -> Statement
reduceTCons (Op If _ T) = T
reduceTCons s = s

reduceFCons :: Statement -> Statement
reduceFCons (Op If p F) = Neg p
reduceFCons s = s

reduceTIff :: Statement -> Statement
reduceTIff (Op Iff T q) = q
reduceTIff (Op Iff p T) = p
reduceTIff s = s

reduceFIff :: Statement -> Statement
reduceFIff (Op Iff F q) = Neg q
reduceFIff (Op Iff p F) = Neg p
reduceFIff s = s

reduceNeg :: Statement -> Statement
reduceNeg (Neg F) = T
reduceNeg (Neg T) = F
reduceNeg s = s

findAllProps :: Statement -> [Statement]
findAllProps (Prop p) = [Prop p]
findAllProps (Op o p q) = unique $ (findAllProps p) ++ (findAllProps q)
findAllProps (Neg p) = unique $ findAllProps p

logicTree :: Statement -> [Statement] -> Tree Statement
logicTree s [] = Node s []
logicTree T _ = Node T []
logicTree F _ = Node F []
logicTree (Prop p) _ = Node (Prop p) [Node T [], Node F []]
logicTree (Neg (Prop p)) _ = Node (Neg (Prop p)) [Node F [], Node T []]
logicTree s (x:xs) =  Node s [(logicTree rt xs), (logicTree rf xs)]
    where rt = recursiveReduce $ replace s x T
          rf = recursiveReduce $ replace s x F

fullLogicTree :: Statement -> Tree Statement
fullLogicTree s = logicTree s $ findAllProps s

replace :: Statement -> Statement -> Statement -> Statement
replace (Op o p q) m r = Op o (replace p m r) (replace q m r)
replace (Neg p) m r = Neg (replace p m r)
replace s m r
    | s == m = r
    | otherwise = s

reduce :: Statement -> Statement
reduce = removeFOr .
         removeTAnd .
         reduceFAnd .
         reduceTOr .
         reduceTAntec .
         reduceFAntec .
         reduceTCons .
         reduceFCons .
         reduceTIff .
         reduceFIff .
         reduceNeg

recursiveReduce :: Statement -> Statement
recursiveReduce (Op o p q) = reduce $ Op o (recursiveReduce p) (recursiveReduce q)
recursiveReduce (Neg p) = reduce $ Neg (recursiveReduce p)
recursiveReduce s = s



