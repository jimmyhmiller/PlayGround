module Main where

import Ast
import AstToJs
import Language.PureScript.Pretty.JS
import Parser


compile :: String -> String
compile = prettyPrintJS . map astToJs . reverse . parse . lexer


main :: IO ()
main = putStr $ compile "\
\fn double(x) { x + x }\
\double(double(2))\
\"
