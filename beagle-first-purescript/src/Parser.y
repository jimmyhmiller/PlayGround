{
module Parser (
  parse,
  lexer
  ) where
import Ast
import Data.Char
}

%name parse
%tokentype { Token }
%error { parseError }

%token
    fn              { TokenFn }
    num             { TokenNum $$ }
    var             { TokenVar $$ }
    '+'             { TokenPlus }
    '-'             { TokenMinus }
    '*'             { TokenTimes }
    '/'             { TokenDiv }
    '('             { TokenOP }
    ')'             { TokenCP }
    '{'             { TokenOCB }
    '}'             { TokenCCB }
    ','             { TokenComma }

%%
    Expr    : Ast                               { [$1] }
            | Expr Ast                          { $2 : $1 }
    Ast     : fn var '(' vars ')' '{' Ast '}'   { Fn (Just $2) $4 $7 }
            | Ast '(' asts ')'                  { App $1 $3 }
            | binOp                             { $1 }
            | Term                              { $1 }
    asts    : Ast                               { [$1] }
            | asts ',' Ast                      { $3 : $1 }
    vars    : var                               { [$1] }
            | vars ',' var                      { $3 : $1 }
    binOp   : Ast '+' Ast                       { BinOp "+" $1 $3 }
    Term    : num                               { Num (Left $1) }
            | var                               { Symbol $1 }

{

getVarName :: Token -> String
getVarName (TokenVar s) = s
getVarName x = error $ "error " ++ show x

parseError :: [Token] -> a
parseError _ = error "Parse error"

data Token
      = TokenFn
      | TokenNum Integer
      | TokenVar String
      | TokenPlus
      | TokenMinus
      | TokenTimes
      | TokenDiv
      | TokenOP
      | TokenCP
      | TokenComma
      | TokenOCB
      | TokenCCB
 deriving Show


lexer :: String -> [Token]
lexer [] = []
lexer (c:cs)
     | isSpace c = lexer cs
     | isAlpha c = lexVar (c:cs)
     | isDigit c = lexNum (c:cs)
lexer ('+':cs) = TokenPlus : lexer cs
lexer ('-':cs) = TokenMinus : lexer cs
lexer ('*':cs) = TokenTimes : lexer cs
lexer ('/':cs) = TokenDiv : lexer cs
lexer ('(':cs) = TokenOP : lexer cs
lexer (')':cs) = TokenCP : lexer cs
lexer (',':cs) = TokenComma : lexer cs
lexer ('{':cs) = TokenOCB : lexer cs
lexer ('}':cs) = TokenCCB : lexer cs

lexNum cs = TokenNum (read num) : lexer rest
     where (num,rest) = span isDigit cs

lexVar cs =
  case span isAlpha cs of
     ("fn",rest) -> TokenFn : lexer rest
     (var,rest)   -> TokenVar var : lexer rest

}
