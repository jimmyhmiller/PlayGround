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
    (x, s) <- parse p2 s
    [(f x, s)])

Monad Parser where
  (>>=) p f = P (\s => do
    (x, s) <- parse p s
    (y, s) <- parse (f x) s
    [(y, s)])


Alternative Parser where
  empty = P (\s => [])
  (<|>) p1 p2 = P (\s => (case parse p1 s of
                               [] => parse p2 s
                               x => x))



satisfy : (pred: Char -> Bool) -> Parser Char
satisfy pred = do
  c <- singleChar
  if (pred c == True)
  then pure c
  else empty
  
digit : Parser Char
digit = satisfy isDigit

space : Parser Char
space = satisfy isSpace

char : Char -> Parser Char
char c = satisfy (== c)


mutual

  oneOrMore : Parser Char -> Parser (List Char)
  oneOrMore p = do
    c <- p
    cs <- zeroOrMore p
    pure (c :: cs)

  
  zeroOrMore : Parser Char -> Parser (List Char)
  zeroOrMore p  = oneOrMore p <|> pure []
  
stringify : Parser (List Char) -> Parser String
stringify p = map pack p

digits : Parser String
digits = stringify (oneOrMore digit)

spaces : Parser String
spaces = stringify (zeroOrMore space)

chars : List Char -> Parser (List Char)
chars [] = pure []
chars (c :: cs) = do
  char c
  chars cs
  pure (c :: cs)


string : String -> Parser String
string s = stringify (chars (unpack s))


token : String -> Parser String
token s = do
  spaces
  string s
  spaces
  pure s

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
    elems <- sepBy (token ",") json
    token "]"
    pure (JArray elems)

  jEntry : Parser (String, Json)
  jEntry = do
    key <- parseString
    token ":"
    value <- json
    pure (key, value)

  jObj : Parser Json
  jObj = do
    token "{"
    entries <- sepBy (token ",") jEntry
    token "}"
    pure (JObj entries)


example1 : String
example1 = "\"test\""

example2 : String
example2 = "{\"test\": 2, \"stuff\": {\"a\": false, \"t\": [1, 2,3]}}"

 
