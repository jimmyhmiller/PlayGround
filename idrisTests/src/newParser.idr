import Effects
import System
import Control.IOExcept
import Effect.System
import Effect.StdIO
import Effect.File

%hide Prelude.List.(++)



data Parser a = Parse (String -> List (a, String))

applyFirst : (a -> b) -> (a, c) -> (b, c)
applyFirst f (a, c) = (f a, c)

parse : Parser a -> String -> List (a, String)
parse (Parse f) y = f y

Functor Parser where 
    map func (Parse f) = Parse (\s => map (applyFirst func) (f s))

Applicative Parser where
    pure x = Parse (\s => [(x, s)])
    (<*>) (Parse f) (Parse g) = Parse (\s => do
      (f, s) <- f s
      (a, s) <- g s
      pure (f a, s))

Monad Parser where
    (>>=) (Parse g) f = Parse (\s => do
      (a, s) <- g s
      (b, s) <- parse (f a) s
      pure (b, s))

Alternative Parser where
    empty = Parse (\s => [])
    (<|>) (Parse f) (Parse g) = Parse (\s => (case f s of
                                                   [] => g s
                                                   res => res))

Semigroup a => Semigroup (Parser a) where
    (<+>) (Parse f) (Parse g) = Parse (\s => f s <+> g s)

item : Parser Char
item = Parse (\s => (case unpack s of
                          [] => empty
                          (x :: xs) => [(x, pack xs)]))

stringify : Parser (List Char) -> Parser String
stringify (Parse f) = Parse (\s => map (applyFirst pack) (f s))

satisfies : (Char -> Bool) -> Parser Char
satisfies p = do
  x <- item
  if p x then pure x else empty


optional : Parser a -> Parser (Maybe a)
optional p = do { x <- p; return (Just x) } <|> return Nothing

mutual
  zeroOrMore : Parser a -> Lazy (Parser (List a))
  zeroOrMore p = oneOrMore p <+> return []

  oneOrMore : Parser a -> Lazy (Parser (List a))
  oneOrMore p = [x::xs | x <- p, xs <- zeroOrMore p]

char : Char -> Parser Char
char x = satisfies (== x)

digit : Parser Char
digit = satisfies isDigit

digits : Parser String
digits = stringify $ oneOrMore digit

lower : Parser Char
lower = satisfies isLower

upper : Parser Char
upper = satisfies isUpper

letter : Parser Char
letter = lower <|> upper

space : Parser Char
space = satisfies isSpace

spaces : Parser String
spaces = stringify $ zeroOrMore space

spaces1 : Parser String
spaces1 = stringify $ oneOrMore space

word : Parser String
word = stringify (oneOrMore letter)

nat : Parser Nat
nat = [cast (pack xs) | xs <- oneOrMore digit]

toListChar : Parser Char -> Parser (List Char)
toListChar p = do
  c <- p
  pure [c]
  
neg : Parser Int
neg = do
  char '-'
  ds <- digits
  pure $ negate (cast ds)
  
pos : Parser Int
pos = do
  ds <- digits
  pure $ cast ds

int : Parser Int
int = neg <|> pos

num : Parser String
num = digits

withSpaces : Parser a -> Parser a
withSpaces parser = spaces *> parser <* spaces

parens : Parser a -> Parser a
parens parser =
 (withSpaces $ char '(')
 *> withSpaces parser
 <* (spaces *> char ')')
 
 
sepBy : (sep : Parser a) -> (parser : Parser b) -> Parser (List b)
sepBy sep parser = do
  frst <- optional parser
  rest <- zeroOrMore (sep *> parser)
  pure $ maybe rest (::rest) frst



fullParse : Parser a -> String -> Maybe a
fullParse (Parse f) s = case f s of
                             [] => Nothing
                             ((a, _) :: xs) => Just a


threewords : Parser (List String)
threewords = do
  word1 <- word
  spaces
  word2 <- word
  spaces
  word3 <- word
  return [word1, word2, word3]
  
  
infixl 3 <||>
(<||>) : Parser a -> Lazy (Parser a) -> Parser a
(<||>) (Parse f) p1 = Parse (\s => (case f s of
                                          [] => parse p1 s
                                          res => res))


symbol : Parser String
symbol = word

data Expr = Symbol String | Num String | Sexpr Expr (List Expr)



mutual
  parseExpr : Parser Expr
  parseExpr = parseSymbol <|> parseNum <||> parseList

  parseList : Parser Expr
  parseList = do 
    char '('
    frst <- parseExpr
    rest <- sepBy spaces1 parseExpr
    char ')'
    return $ Sexpr frst rest
  
  parseSymbol : Parser Expr
  parseSymbol = map Symbol (symbol <|> stringify (toListChar $ char '+'))
  
  parseNum : Parser Expr
  parseNum =  map Num num

parseLisp : Parser (List Expr)
parseLisp = oneOrMore (withSpaces parseExpr)


seperated : (sep : String) -> (coll : List String) -> String
seperated sep [] = ""
seperated sep [x] = x
seperated sep (x :: xs) = x ++ sep ++ seperated sep xs

commaSeparated : (coll : List String) -> String
commaSeparated = seperated ", "

newLineSeparated : (coll : List String) -> String
newLineSeparated = seperated "\n"

indention : Nat -> String
indention k = concat $ replicate k "  "







 
data JsExpr = JsNum String | JsSymbol String | JsFunction JsExpr (List JsExpr) (List JsExpr) JsExpr | JsApplication JsExpr (List JsExpr) | JsBinOp JsExpr JsExpr JsExpr



lispToJs : Expr -> JsExpr
lispToJs (Symbol x) = JsSymbol x
lispToJs (Num x) = JsNum x
lispToJs (Sexpr (Symbol "defn") (Symbol name :: (Sexpr arg args) :: return :: [])) = JsFunction (lispToJs (Symbol name)) (map lispToJs (arg :: args)) [] (lispToJs return)
lispToJs (Sexpr (Symbol "+") (left :: right :: [])) = JsBinOp (lispToJs (Symbol "+")) (lispToJs left) (lispToJs right)
lispToJs (Sexpr x ys) = JsApplication (lispToJs x) (map lispToJs ys)


isSpaceOrEmpty : String -> Bool
isSpaceOrEmpty "" = True
isSpaceOrEmpty s =  all isSpace $ unpack s

removeEmpty : List String -> List String
removeEmpty = filter (not . isSpaceOrEmpty)

toJs : (indent : Nat) -> JsExpr -> String
toJs indent (JsNum x) = x
toJs indent (JsSymbol x) = x
toJs indent (JsFunction (JsSymbol name) args body return) = unlines $ removeEmpty [
  "const " ++ name ++ " = function(" ++ (commaSeparated (map (toJs indent) args)) ++ ") {",
    indention indent ++ unlines (map (toJs (S indent)) body),
    indention indent ++ "return " ++ (toJs (S indent) return),
  "}"]
toJs indent (JsApplication fn args) = (toJs indent fn) ++ "(" ++ (commaSeparated (map (toJs indent) args)) ++ ")"
toJs indent (JsBinOp op left right) = (toJs indent left) ++ " " ++ (toJs indent op) ++ " " ++ (toJs indent right)



compile : String -> Maybe String
compile s = do
  lisp <- fullParse parseLisp s
  let jsExprs = map lispToJs lisp
  let js = map (toJs 1) jsExprs
  pure $ unlines js
  

double : Maybe String
double = compile """
(defn double (x) 
  (+ x x))
(double 2)
(defn identity (x)
  x)
(identity 3)
"""

printCompile : Either String (Maybe String) -> IO ()
printCompile (Left l) = putStr' l
printCompile (Right Nothing) = return ()
printCompile (Right (Just x)) = putStr' x

printCompile' : Maybe String -> IO ()
printCompile' Nothing = return ()
printCompile' (Just x) = putStr' x


--emain : Eff (Either String String) [SYSTEM, STDIO, FILE_IO ()]
--emain = do [prog, file] <- getArgs
--           pure $ Right  


main : IO ()
main = printCompile' (   compile """
(defn double (x) (+ x x))
(double 2)
(defn identity (x) x)
(identity 2)
""")

 
