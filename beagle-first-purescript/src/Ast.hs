module Ast
    (
      Ast(..)
    ) where

data Ast = Fn (Maybe String) [String] Ast
         | App Ast [Ast]
         | BinOp String Ast Ast
         | Num (Either Integer Double)
         | Symbol String
         deriving (Eq, Show)
