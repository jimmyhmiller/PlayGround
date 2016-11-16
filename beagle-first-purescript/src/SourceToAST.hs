module SourceToAst
    (
    parser
    ) where

import Debug.Trace(trace)
import Text.Parsec
import Text.ParserCombinators.Parsec.Prim
import Text.ParserCombinators.Parsec.Char
import Ast
import Data.Functor.Identity


oneOrMore :: Parser a -> Parser [a]
oneOrMore = many1

zeroOrMore :: Parser a -> Parser [a]
zeroOrMore = many

word :: Parser String
word = oneOrMore letter

lit :: String -> Parser String
lit = string

withSpaces :: Parser a -> Parser a
withSpaces parser = zeroOrMore space *> parser <* zeroOrMore space

neg :: Parser Integer
neg = do
  _ <- char '-'
  ds <- many1 digit
  pure $ negate (read ds)

pos :: Parser Integer
pos = do
  ds <- many1 digit
  pure $ read ds

int :: Parser Integer
int = neg <|> pos

parser :: Parser [Ast]
parser = oneOrMore $ withSpaces parseAst


parseAst :: Parser Ast
parseAst = fn
       <|> Text.Parsec.try (num >>= \n -> binOp n)
       <|> Text.Parsec.try (symbol >>= \n -> binOp n)
       <|> Text.Parsec.try (symbol >>= \s -> app s)
       <|> symbol
       <|> num

symbol :: Parser Ast
symbol = fmap Symbol word

fn :: Parser Ast
fn = do
  lit "fn"
  fnName <- withSpaces $ optionMaybe word
  lit "("
  args <- withSpaces $ zeroOrMore word
  lit ")"
  withSpaces $ lit "{"
  body <- parseAst
  withSpaces $ lit "}"
  return $ Fn fnName args body

app :: Ast -> Parser Ast
app fn = do
  lit "("
  args <- zeroOrMore parseAst
  lit ")"
  return $ App fn args

num :: Parser Ast
num = do
  n <- int
  return $ Num (Left n)
binOp :: Ast -> Parser Ast
binOp left = do
  op <- withSpaces $ oneOf "*/-+"
  right <- parseAst
  return $ BinOp [op] left right
