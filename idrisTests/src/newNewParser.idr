

data Parser a = Parse (String -> List (a, String))


parse : Parser a -> String -> List (a, String)
parse (Parse f) y = f y


parse_single_char : Parser Char
parse_single_char = Parse (\s => (case unpack s of
                                       [] => []
                                       (x :: xs) => [(x, pack xs)]))

apply_first : (a -> b) -> (a, c) -> (b, c)
apply_first f (a, b) = (f a, b)


Functor Parser where
  map func p = Parse (\s => map (apply_first func) (parse p s))


Applicative Parser where
  pure x = Parse (\s => [(x, s)])
  (<*>) p1 p2 = Parse
    (\s => do
      (f, s) <- parse p1 s
      (a, s) <- parse p2 s
      pure (f a, s))


Monad Parser where
  (>>=) p1 f = Parse (\s => do
    (a, s) <- parse p1 s
    p2 <- parse (f a) s
    pure p2)


Alternative Parser where

  empty = Parse (\s => [])
  (<|>) p1 p2 = Parse (\s =>
    (case parse p1 s of
      [] => parse p2 s
      res => res))
     

satisfies : (Char -> Bool) -> Parser Char
satisfies pred = do
  c <- parse_single_char
  if (pred c)
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


stringify : Parser (List Char) -> Parser String
stringify p = map pack p
 
 
mutual
  zeroOrMore : Parser a -> Lazy (Parser (List a))
  zeroOrMore p = oneOrMore p <|> pure []
  
  oneOrMore : Parser a -> Lazy (Parser (List a))
  oneOrMore p = do
    x <- p
    xs <- zeroOrMore p
    pure (x :: xs)


spaces : Parser String
spaces = stringify $ zeroOrMore space

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

withSpaces : Parser a -> Parser a
withSpaces p = do
  spaces
  res <-p
  spaces
  pure res

parens : Parser a -> Parser a
parens p = do
  withSpaces $ char '('
  res <- withSpaces p
  withSpaces $ char ')'
  pure res

exactlyOne : Parser a -> Parser (List a)
exactlyOne p = do
  res <- p
  pure [res]

sepBy : (sep : Parser a) -> (p : Parser b) -> Parser (List b)
sepBy sep p = (do
  elem <- p
  sep
  rest <- sepBy sep p
  pure $ elem :: rest
  )
  <|> exactlyOne p
  
  
leftAssoc : (p : Parser a) -> (op : Parser (a -> a -> a)) -> Parser a
leftAssoc p op = do {
    a <- p
    rest a
  }
  where 
    rest a = (do
      f <- op
      b <- p
      rest (f a b)
    ) <|> pure a

data Expr = Number Int | Plus Expr Expr | Subtract Expr Expr


parseNumber : Parser Expr
parseNumber = do
  n <- int
  pure $ Number n

plus : Parser (Expr -> Expr -> Expr)
plus = do 
  char '+'
  pure $ Plus
  
sub : Parser (Expr -> Expr -> Expr)
sub = do 
  char '-'
  pure $ Subtract
  
op : Parser (Expr -> Expr -> Expr)
op = plus <|> sub


parseExpr : Parser Expr
parseExpr = parseNumber `leftAssoc` op
