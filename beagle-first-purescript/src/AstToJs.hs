module AstToJs
    (
      astToJs
    ) where
import Language.PureScript.CodeGen.JS.AST
import Ast

toJSBinary :: String -> BinaryOperator
toJSBinary "+" = Add
toJSBinary "-" = Subtract
toJSBinary "*" = Multiply
toJSBinary "/" = Divide

astToJs :: Ast -> JS
astToJs (Fn name args body) = JSFunction Nothing name args (JSBlock Nothing [JSReturn Nothing (astToJs body)])
astToJs (App fn args) = JSApp Nothing (astToJs fn) (map astToJs args)
astToJs (BinOp op left right) = JSBinary Nothing (toJSBinary op) (astToJs left) (astToJs right)
astToJs (Num n) = JSNumericLiteral Nothing n
astToJs (Symbol s) = JSVar Nothing s
