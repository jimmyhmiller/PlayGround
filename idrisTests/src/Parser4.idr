
data Parser a = P (String -> List (a, String))

parse : Parser a -> String -> List (a, String)
parse (P f) y = f y


single_char : Parser Char
single_char = P (\s => case unpack s of
                            [] => []
                            (x :: xs) => [(x, pack xs)])


apply_first : (a -> b) -> (a, String) -> (b, String)
apply_first f (a, b) = (f a, b)

Functor Parser where
  map func p = P (\s => map (apply_first func) (parse p s))

Applicative Parser where
  pure x = P (\s => [(x, s)])
  (<*>) p1 p2 = P (\s => do
    (f, s) <- parse p1 s
    (a, s) <- parse p2 s
    pure (f a, s))

Monad Parser where
  (>>=) p f = P (\s => do
    (a, s) <- parse p s
    (b, s) <- parse (f a) s 
    pure (b, s))


Alternative Parser where
  empty = P (\s => [])
  (<|>) p1 p2 = P (\s => 
    case parse p1 s of
      [] => parse p2 s
      ret => ret)

satisfies : (pred: Char -> Bool) -> Parser Char
satisfies pred = do
  c <- single_char
  if pred c
    then pure c
    else empty
    

char : Char -> Parser Char
char x = satisfies (== x)

digit : Parser Char
digit = satisfies isDigit

lower : Parser Char
lower = satisfies isLower

upper : Parser Char
upper = satisfies isUpper

letter : Parser Char
letter = lower <|> upper

space : Parser Char
space = satisfies isSpace

mutual
  zeroOrMore : Parser a -> Parser (List a)
  zeroOrMore p = oneOrMore p <|> pure []
  
  oneOrMore : Parser a -> Parser (List a)
  oneOrMore p = do
    x <- p
    xs <- zeroOrMore p
    pure (x::xs)
  

stringify : Parser (List Char) -> Parser String
stringify p = map pack p

spaces : Parser String
spaces = stringify (zeroOrMore space)

spaces1 : Parser String
spaces1 = stringify $ oneOrMore space

digits : Parser String
digits = stringify $ oneOrMore digit

word : Parser String
word = stringify $ oneOrMore letter


nat : Parser Nat
nat = do
  xs <- digits
  pure $ cast xs

neg : Parser Int
neg = do
  char '-'
  num <- digits
  pure $ (negate (cast num))

pos : Parser Int
pos = do
  num <- digits
  pure $ cast num
 
int : Parser Int
int = neg <|> pos
