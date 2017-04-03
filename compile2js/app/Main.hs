{-# LANGUAGE OverloadedStrings #-}
module Main where

import Data.Aeson as A
import qualified Data.Text as T
import Data.Scientific
import GHC.Exts
import Data.ByteString.UTF8

import qualified Data.Text.Lazy.IO as DTL
import qualified Data.Text.Lazy.Encoding as DTL


data Program = Program [Statement] deriving Show

data Statement = ExpressionStatement Expr
               | BlockStatement [Statement]
               | ReturnStatement (Maybe Expr)
   deriving Show

data Expr = Lit Literal
          | Binary BinaryOperator Expr Expr
          | Identifier T.Text
          | Assign AssignmentOperator Expr Expr
          | FunctionExpression (Maybe Expr) [Expr] Statement
   deriving Show

data BinaryOperator = EQUAL | NEQUAL | ADD | SUB | MULT | DIV


instance Show BinaryOperator where
  show EQUAL = "==="
  show NEQUAL = "!=="
  show ADD = "+"
  show SUB = "-"
  show MULT = "*"
  show DIV = "/"

data AssignmentOperator = SETEQUAL

instance Show AssignmentOperator where
  show SETEQUAL = "="

data Node = Node String (Maybe SourceLocation) deriving Show

data SourceLocation = SourceLocation String Position Position deriving Show

data Position = Position Int Int deriving Show

data Literal = S T.Text | B Bool | Null | Number Scientific deriving Show

expr :: Expr -> Statement
expr = ExpressionStatement

num :: Scientific -> Expr
num a = Lit (Main.Number a)

add :: Scientific -> Scientific -> Expr
add a b = Binary ADD (num a) (num b)

(.=.) :: Expr -> Expr -> Expr
(.=.) = Assign SETEQUAL

(.+.) :: Expr -> Expr -> Expr
(.+.) = Binary ADD

lit :: Literal -> Statement
lit l = ExpressionStatement (Lit l)

instance ToJSON Statement where
  toJSON (ExpressionStatement e) = object [
    "type" .= String "ExpressionStatement",
    "expression" .= toJSON e ]
  toJSON (BlockStatement xs) = object [
    "type" .= String "BlockStatement",
    "body" .= Array (fromList $ map toJSON xs)]
  toJSON (ReturnStatement e) = object [
    "type" .= String "ReturnStatement",
    "argument" .= toJSON e]

jsReturn = ReturnStatement

instance ToJSON BinaryOperator where
  toJSON o = String $ T.pack $ show o

instance ToJSON AssignmentOperator where
  toJSON o = String $ T.pack $ show o


operator t op left right = object [
  "type" .= String t,
  "operator" .= toJSON op,
  "left" .= toJSON left,
  "right"  .= toJSON right]


instance ToJSON Expr where
  toJSON (Lit l) = toJSON l
  toJSON (Binary op left right) = operator "BinaryExpression" op left right
  toJSON (Identifier s) = object [
    "type" .= String "Identifier",
    "name" .= String s]
  toJSON (Assign op left right) = operator "AssignmentExpression" op left right
  toJSON (FunctionExpression ident params body) = object [
    "type" .= String "FunctionExpression",
    "id" .= toJSON ident,
    "params" .= Array (fromList $ map toJSON params),
    "body" .= toJSON body]


lambda params body = FunctionExpression Nothing params $ block [jsReturn body]

typeValue t v = object [
  "type" .= String t,
  "value" .= v ]

instance ToJSON Literal where
  toJSON (S s) = typeValue "Literal" $ String s
  toJSON (B b) = typeValue "Literal" $ Bool b
  toJSON Main.Null = typeValue "Literal" A.Null
  toJSON (Main.Number n) = typeValue "Literal" $ A.Number n

instance ToJSON Program where
  toJSON (Program xs) = object [
    "type" .= String "Program",
    "body" .= Array (fromList $ map toJSON xs)]


i = Identifier

block = BlockStatement

identityFunc = lambda [i "x"] $ return $ i "x"

addem = [expr $ i "x" .=. add 2 2, expr $ i "x" .+. Identifier "x"]

main :: IO ()
main = DTL.putStr $ DTL.decodeUtf8 $ encode $ Program [expr $ i "identity" .=. identityFunc]
