
data Parser a = Parse (String -> List (a, String))



parse : Parser a -> String -> List (a, String)
parse (Parse f) s = f s


parse_single_char : Parser Char
parse_single_char = Parse (\s => (case unpack s of
                                       [] => []
                                       (x :: xs) => [(x, pack xs)]))

apply_first : (a -> b) -> (a, c) -> (b, c)
apply_first f (a, c) = (f a, c)

Functor Parser where
  map f p = Parse (\s => map (apply_first f) (parse p s))

Applicative Parser where
  pure x = Parse (\s => [(x, s)])
  (<*>) p1 p2 = Parse (\s => do
    (f, s) <- parse p1 s
    (a, s) <- parse p2 s
    pure (f a, s))


Monad Parser where
  (>>=) p f = Parse(\s => do
    (a, s) <- parse p s
    y <- parse (f a) s
    pure y)


Alternative Parser where
  empty = Parse(\s => [])
  (<|>) p1 p2 = Parse(\s => do
    (case parse p1 s of
      [] => parse p2 s
      res => res))
    

satisfies : (pred: Char -> Bool) -> Parser Char
satisfies pred = do
  c <- parse_single_char
  if (pred c)
    then pure c
    else empty


char : Char -> Parser Char
char c = satisfies (== c)

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
  zeroOrMore : Parser a -> Lazy (Parser (List a))
  zeroOrMore p = oneOrMore p <|> pure []
  
  oneOrMore : Parser a -> Lazy (Parser (List a))
  oneOrMore p = do
    x <- p
    xs <- zeroOrMore p
    pure (x :: xs)


stringify : Parser (List Char) -> Parser String
stringify p = do
  c <- p
  pure $ pack c


spaces : Parser String
spaces = stringify $ zeroOrMore space
