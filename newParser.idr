data Parser a = Parse (String -> List (a, String))

applyFirst : (a -> b) -> (a, c) -> (b, c)
applyFirst f (a, b) = (f a, b)

parse : Parser a -> String -> List (a, String)
parse (Parse f) y = f y

Functor Parser where
    map func (Parse f) = Parse (\s => map (applyFirst func) (f s))

Applicative Parser where
    pure x = Parse (\s => [(x, s)])
    (<*>) (Parse f) (Parse g) = Parse (\s => do 
     (func, s1) <- f s
     (a, s2) <- g s1
     return (func a, s2))

Monad Parser where
    (>>=) (Parse g) f = Parse (\s => do 
       (item, s1) <- g s
       parse (f item) s1)

Alternative Parser where
    empty = Parse (\s => [])
    (<|>) (Parse f) y = ?Alternative_rhs_1


item : Parser Char
item = Parse (\inp => (case unpack inp of
                          [] => []
                          (x :: xs) => [(x, pack xs)]))


sat : (Char -> Bool) -> Parser Char                                                               
sat p = do 
  x <- item
  if p x then pure x else empty
  
char : Char -> Parser Char
char x = sat (== x)

digit : Parser Char
digit = sat isDigit

lower : Parser Char
lower = sat isLower

upper : Parser Char
upper = sat isUpper
