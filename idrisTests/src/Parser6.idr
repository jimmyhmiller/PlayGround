data Parser a = P (String -> List (a, String))

parse : Parser a -> String -> List (a, String)
parse (P f) s = f s


singleChar : Parser Char
singleChar = P (\s => case unpack s of
                        [] => []
                        (x :: xs) => [(x, pack xs)])


applyFirst : (a -> b) -> (a, String) -> (b, String)
applyFirst f (a, s) = (f a, s)

Functor Parser where
  map func p = P (\s => map (applyFirst func) (parse p s))

Applicative Parser where
  pure x = P (\s => [(x, s)])
  (<*>) p1 p2 = P (\s => do
    (f, s) <- parse p1 s
    (a, s) <- parse p2 s
    pure $ (f a, s))

Monad Parser where
  (>>=) p f = P (\s => do
    (x, s) <- parse p s
    (y, s) <- parse (f x) s
    pure $ (y, s))

Alternative Parser where
  empty = P (\s => [])
  (<|>) p1 p2 = P (\s => case parse p1 s of
                      [] => parse p2 s
                      ret => ret)

satisfy : (Char -> Bool) -> Parser Char
satisfy pred = do
  c <- singleChar
  if (pred c)
  then pure c
  else empty

char : Char ->  Parser Char
char c = satisfy (== c)

digit : Parser Char
digit = satisfy isDigit

space : Parser Char
space = satisfy isSpace


mutual
  zeroOrMore : Parser a -> Parser (List a)
  zeroOrMore p = (oneOrMore p) <|> pure []
  
  oneOrMore : Parser a -> Parser (List a)
  oneOrMore p = do
    x <- p
    xs <- zeroOrMore p
    pure (x :: xs)
    
    
stringify : Parser (List Char) -> Parser String
stringify p = map pack p

spaces : Parser String
spaces = stringify $ zeroOrMore space

digits : Parser String
digits = stringify $ oneOrMore digit


chars : List Char -> Parser (List Char)
chars [] = pure []
chars (c :: cs) = do
  char c
  chars cs
  pure (c :: cs)

string : String -> Parser String
string s = stringify $ chars (unpack s)

token : String -> Parser String
token s = do
  spaces
  x <- string s
  spaces
  pure x

sepBy : (sep : Parser a) -> (p : Parser b) -> Parser (List b)
sepBy sep p = (do
    elem <- p
    sep
    rest <- sepBy sep p
    pure $ elem :: rest
  )
  <|> map pure p


untilC : Char -> Parser String
untilC c = stringify $ zeroOrMore $ satisfy (/= c)

parseString : Parser String
parseString = do
  char '"'
  xs <- untilC '"'
  char '"'
  pure xs

true : Parser Bool
true = do
  string "true"
  pure True
  
false : Parser Bool
false = do
  string "false"
  pure False

bool : Parser Bool
bool = true <|> false

int : Parser Int
int = do
  sign <- string "-" <|> pure ""
  xs <- digits
  pure $ cast (sign ++ xs)

data Json
  = JString String
  | JBool Bool
  | JNum Int
  | JArray (List Json)
  | JObj (List (String, Json))


mutual
  json : Parser Json
  json = jString <|> jNum <|> jBool <|> jArray <|> jObj

  jBool : Parser Json
  jBool = map JBool bool

  jString : Parser Json
  jString = map JString parseString

  jNum : Parser Json
  jNum = map JNum int

  jArray : Parser Json
  jArray = do
    token "["
    xs <- sepBy (token ",") json
    token "]"
    pure $ JArray xs

  jEntry : Parser (String, Json)
  jEntry = do
    key <- parseString
    token ":"
    value <- json
    pure $ (key, value)

  jObj : Parser Json
  jObj= do
    token "{"
    entries <- sepBy (token ",") jEntry
    token "}"
    pure $ JObj entries


example1 : String
example1 = "\"test\""

example2 : String
example2 = "{\"test\": 2, \"stuff\": {\"a\": false, \"t\": [1, 2,3]}}"

