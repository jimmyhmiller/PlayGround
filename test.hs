pascal = iterate (\row -> zipWith (+) ([0] ++ row) (row ++ [0])) [1]

main = print $ take 10 pascal