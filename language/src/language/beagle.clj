(ns language.beagle
  (:require [instaparse.core :as insta]))



(def parser
  (insta/parser
   "
<program> = data* | fn* | fn-application* | val* | infix-application* | program*
data = <'data'> data-name <'='> data-body
data-name = upper-name
<upper-name> = #'[A-Z][a-zA-Z]*'
<data-body> = constructor (<'|'> constructor)*
constructor = constructor-name args | constructor-name | constructor-name keywords
keywords = <'('> <'{'> (keyword <','?>)* <'}'> <')'>
keys = <'('> <'{'> (arg <','?>)* <'}'> <')'>
args = <'('> (arg <','?>)* <')'> | arg | literal
fn-args = <'('> ((arg | deconstructor)  <','?>)* <')'> | arg | literal | deconstructor
<constructor-name> = upper-name
<arg> = identifier
<keyword> = #':[A-Za-z]+'
identifier = #'[A-Za-z\\-_!]+'
literal = #'[0-9]' | keyword
<expr> = fn-application | literal | identifier | infix-application | lambda | val
fn-application = identifier <'('> (expr <','?>)* <')'>
infix-application = expr <' '> symbol <' '> expr
symbol = #'[$-/:-?{-~!\"^_`\\[\\]]'

deconstructor = constructor-name | constructor-name args | constructor-name keys

val = <'val'> identifier <'='> expr

fn = <'fn'> identifier <'{'> fn-body <'}'> | <'fn'> identifier lambda
fn-body = lambda+
lambda = fn-args  <'=>'> expr | fn-args <'=>'> <'{'> expr+ <'}'>
"
    :auto-whitespace :standard))


(defn clojure-parse [string]
  (->> (parser string)
       (insta/transform
        {:data (fn [data-name & data-bodies] [(keyword data-name) data-bodies])
         :data-name identity
         :fn (fn [fn-name fn-body] `(defn ~fn-name ~@fn-body))
         :lambda (fn [args body] `([~@args] ~body)) 
         :fn-args (fn [& args]  args)
         :infix-application (fn [left op right] `(~op ~left ~right))
         :identifier (fn [name] (symbol name))
         :literal (fn [x] (read-string x))
         :symbol (fn [sym] (symbol sym))
         :fn-application (fn [fn-name & args] 
                           `(~fn-name  ~@args))})))




(def prog "fn double(x) => { x * 2 }")

(parser prog)
(clojure-parse prog)
