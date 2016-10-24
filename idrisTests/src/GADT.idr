data Symbol = Sym String
data Number = Num String


data JSExpr : Type -> Type where
  JSSymbol : Symbol -> JSExpr Symbol
  JSNumber : Number -> JSExpr Number
  JSFunction : JSExpr Symbol -> List (JSExpr Symbol) -> JSExpr a -> JSExpr ()


num : String -> JSExpr Number
num s = JSNumber (Num s)

sym : String -> JSExpr Symbol
sym s = JSSymbol (Sym s)
