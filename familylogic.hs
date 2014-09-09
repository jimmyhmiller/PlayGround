import Data.List (nub)
import Data.Tree
import Data.Tree.Pretty

unique = nub

data Relation = Parent String String
              | Spouse String String deriving (Eq, Show)

type Db = [Relation]

makeChildren :: String -> String -> [String] -> [Relation]
makeChildren _ _ [] = []
makeChildren d m (x:xs) = (Spouse d m):(Parent d x):(Parent m x):makeChildren d m xs

spouse :: String -> Relation -> Bool
spouse s (Spouse s' s'') = s == s' || s == s''
spouse _ _ = False

getSpouse :: String -> Relation -> String
getSpouse s (Spouse s' s'') 
    | s == s' = s''
    | otherwise = s'

spouses :: Db -> String -> [String]
spouses db p = map (getSpouse p) $ filter (spouse p) db

child :: String -> Relation -> Bool
child p (Parent p' _) = p == p'
child _ _ = False

parent :: String -> Relation -> Bool
parent c (Parent _ c') = c == c'
parent _ _ = False

parents :: Db -> String -> [String]
parents rs c = map getParent $ filter (parent c) rs

grandparents :: Db -> String -> [String]
grandparents db p = concatMap (parents db) (parents db p) 

siblings ::  Db -> String -> [String]
siblings db p = filter (/= p) $ unique $ concatMap (children db) $ parents db p

auntsOrUncles :: Db -> String -> [String]
auntsOrUncles db p = concatMap (siblings db) $ parents db p

nauntsOrUncles :: Db -> String -> Int -> [String]
nauntsOrUncles db p n = concatMap (auntsOrUncles db) $ nparents db p n

cousins :: Db -> String -> [String]
cousins db p = concatMap (children db) $ auntsOrUncles db p

ncousins :: Db -> String -> Int -> [String]
ncousins db p 2 = concatMap (children db) $ cousins db p
ncousins db p n = concatMap (children db) $ ncousins db p (n-1)

nparents :: Db -> String -> Int -> [String]
nparents db p 1 = parents db p
nparents db p n = concatMap (parents db) (nparents db p (n-1))

nchildren :: Db -> String -> Int -> [String]
nchildren db p 1 = children db p
nchildren db p n = concatMap (children db) (nchildren db p (n-1))

cousinsNRemoved :: Db -> String -> Int -> [String]
cousinsNRemoved db p n = concatMap (cousins db) $ (nparents db p n ++ nchildren db p n)

getChild :: Relation -> String
getChild (Parent _ c) = c

getParent :: Relation -> String
getParent (Parent p _) = p

children :: Db -> String -> [String]
children rs p = map getChild $ filter (child p) rs

buildTree :: Db -> String -> Tree String
buildTree db p = Node p (map (buildTree db) $ children db p)

db = makeChildren "Fred Jr" "Cindy" ["Jimmy", "Carrie", "Fred III"] ++
     makeChildren "Herbie" "Ginny" ["Cindy", "Tim", "Terry", "Mark"] ++
     makeChildren "Terry" "Terra" ["Jacyln", "Courtney", "Tucker"] ++
     makeChildren "GreatF" "GreatM" ["Herbie", "Joe", "Mary"] ++
     makeChildren "Fredrick" "Loreine" ["Francine", "Keith", "Kenny", "Angel", "Fred Jr"] ++
     makeChildren "Francine" "Gary Wayne" ["Zane", "Cory"] ++
     makeChildren "Zane" "Kacie" ["Kinley", "Zyler"] ++
     makeChildren "Angel" "Everett Trask IV" ["Everett V", "Eric", "Erin"] ++
     makeChildren "Everett V" "Cristina" ["Siena"] ++
     makeChildren "Erin" "Amber" ["Jayden", "Aubrey"] ++
     makeChildren "Keith" "Coni" ["Breannan", "Natalee", "Dakota"] ++
     makeChildren "Kenny" "Kenny's wife" ["Rachel"]



main = putStrLn $ drawVerticalTree $ buildTree db "Fredrick"