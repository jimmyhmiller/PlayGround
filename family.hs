import Data.Tree
import Data.Tree.Pretty
import Data.Tree.Zipper
import Control.Applicative
import Control.Monad

data People = Single String | Couple String String deriving (Eq)


type FamilyTree = Tree People

instance Show People where
    show (Single p) = p
    show (Couple w h) = w ++ " and " ++ h


addParents :: [FamilyTree] -> FamilyTree -> FamilyTree
addParents c (Node (Couple m d) []) = Node (Couple m d) c
addParents c (Node (Couple m d) cs) = Node (Couple m d) (c ++ cs)

addSpouse :: FamilyTree -> FamilyTree -> FamilyTree
addSpouse (Node (Single w) []) (Node (Single h) []) = Node (Couple w h) []

addChildren :: FamilyTree -> [FamilyTree] -> FamilyTree
addChildren = flip addParents



tim = Node (Couple "Tim" "Patty") [Node (Single "Timi") []]
terry = Node (Couple "Terry" "Terra") [Node (Single "Courtney") [], Node (Single "Jaclyn") [], Node (Single "Tucker") []]
mark = Node (Single "Mark") []
cindy = Node (Couple "Cindy" "Fred") [Node (Couple "Jimmy" "Janice") [], Node (Couple "Carrie" "Austin") [], Node (Couple "Fred" "Emily") []]
family = Node (Couple "Herby" "Ginny") [tim, terry, mark, cindy]



draw :: Show a => Tree a -> String
draw = drawVerticalTree . fmap show


main = putStrLn $ draw family