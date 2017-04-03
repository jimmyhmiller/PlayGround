{-# LANGUAGE DataKinds, TypeOperators, KindSignatures, TypeFamilies, TemplateHaskell #-}
{-# LANGUAGE FlexibleContexts, NoMonomorphismRestriction, GADTs, FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}

module Records where

import Data.Vinyl
import Data.Vinyl.Functor
import Control.Applicative
import Control.Lens hiding (Identity, (.=))
import Control.Lens.TH
import Data.Char
import Data.Singletons.TH
import Data.Maybe
import Data.Scientific
import Data.Text
import qualified Data.Aeson as A
import Data.Aeson as A ((.=), object)
import qualified Data.Text.Lazy.IO as DTL
import qualified Data.Text.Lazy.Encoding as DTL

data Fields = Type
            | Loc
            | Source
            | Start
            | End
            | Line
            | Column
            | Elements
            | Name
            | Value
            | Body
            | Expr
  deriving Show

data Lit = String Text
         | Boolean Bool
         | Null
         | Number Scientific

instance Show Lit where
  show (String t) = show t
  show (Boolean b) = show b
  show (Null) = "null"
  show (Number n) = show n

type Node = [Type, Loc]
type SourceLocation = [Source, Start, End]
type Position = [Line, Column]
type Expression = Node
type Statement = Node
type Array = [Elements, Type, Loc]
type Identifier = [Name, Type, Loc]
type Literal = [Value, Type, Loc]
type Program = [Body, Type, Loc]
type ExpressionStatement = [Expr, Type, Loc]

type family ElF (f :: Fields) :: * where
  ElF Type = Text
  ElF Loc = Maybe (Rec Attr SourceLocation)
  ElF Source = Maybe String
  ElF Start = Rec Attr Position
  ElF End = Rec Attr Position
  ElF Line = Scientific
  ElF Column = Scientific
  ElF Elements = [Maybe (Rec Attr Expression)]
  ElF Name = Text
  ElF Value = Lit
  ElF Body = [Rec Attr Statement]
  ElF Expr = Rec Attr Expression

newtype Attr f = Attr { _unAttr :: ElF f }
makeLenses ''Attr
genSingletons [ ''Fields ]

instance Show (Attr Type) where show (Attr x) = "type: " ++ show x
instance Show (Attr Loc) where show (Attr x) = "loc: " ++ show x
instance Show (Attr Source) where show (Attr x) = "source: " ++ show x
instance Show (Attr Start) where show (Attr x) = "start: " ++ show x
instance Show (Attr End) where show (Attr x) = "end: " ++ show x
instance Show (Attr Line) where show (Attr x) = "line: " ++ show x
instance Show (Attr Column) where show (Attr x) = "column: " ++ show x
instance Show (Attr Elements) where show (Attr x) = "elements: " ++ show x
instance Show (Attr Name) where show (Attr x) = "name: " ++ show x
instance Show (Attr Value) where show (Attr x) = "value: " ++ show x
instance Show (Attr Body) where show (Attr x) = "body: " ++ show x
instance Show (Attr Expr) where show (Attr x) = "expr: " ++ show x

instance A.ToJSON (Rec Attr Node) where
  toJSON r = object [
    "type" .= A.toJSON (r .^. SType),
    "loc" .= A.toJSON (r .^. SLoc)]

instance A.ToJSON (Rec Attr SourceLocation) where
  toJSON r = object [
    "source" .= A.toJSON (r .^. SSource),
    "start" .= A.toJSON (r .^. SStart),
    "end" .= A.toJSON (r .^. SEnd)]

instance A.ToJSON (Rec Attr Position) where
  toJSON r = object [
    "line" .= A.toJSON(r .^. SLine),
    "column" .= A.toJSON (r .^. SColumn)]

instance A.ToJSON (Rec Attr Program) where
  toJSON r = object [
    "body" .= A.toJSON (r .^. SBody),
    "type" .= A.toJSON (r .^. SType),
    "loc" .= A.toJSON (r .^. SLoc)]

instance A.ToJSON (Rec Attr ExpressionStatement) where
  toJSON r = object[
    "expression" .= A.toJSON (r .^. SExpr),
    "type" .= A.toJSON (r .^. SType),
    "loc" .= A.toJSON (r .^. SLoc)]

blankNode :: Rec Attr Node
blankNode = SType =:: "" :& SLoc =:: Nothing :& RNil

blankIdentifier :: Rec Attr Identifier
blankIdentifier = SName =:: "" :& blankNode

blankLiteral :: Rec Attr Literal
blankLiteral = SValue =:: Null :& blankNode

blankExprState :: Rec Attr ExpressionStatement
blankExprState = rput (SType =:: "ExpressionStatement") $ (SExpr =:: blankNode) :& blankNode

thisExpr :: Rec Attr Expression
thisExpr = rput (SType =:: "ThisExpression") blankNode

functionExpr :: Rec Attr Expression
functionExpr = rput (SType =:: "FunctionExpression") blankNode

identifier :: Text -> Rec Attr Identifier
identifier name = rput (SType =:: "Identifier") $ rput (SName =:: name) blankIdentifier

literal :: Lit -> Rec Attr Literal
literal val = rput (SType =:: "Literal") $ rput (SValue =:: val) blankLiteral

exprState :: Rec Attr Expression -> Rec Attr ExpressionStatement
exprState e = rput (SExpr =:: e) blankExprState

(.^.) r attr = r ^. rlens attr . unAttr


getType :: (Type ∈ rs) => Rec Attr rs -> Text
getType r = r ^. rlens SType . unAttr




(=::) :: sing f -> ElF f -> Attr f
_ =:: x = Attr x

node = (SType =:: "test")
       :& RNil


main :: IO ()
main = print $ exprState (rcast $ literal (String "hello"))
