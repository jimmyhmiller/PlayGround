(ns try-language.core
  (:require [om.core :as om :include-macros true]
            [om.dom :as dom :include-macros true]
            [instaparse.core :as insta])
  (:use-macros [try-language.macros :only [defclass]]))



(def parser
  (insta/parser
  "<program> = class* | variable* | expression* | program*
   class = <'class'> <' '> class-name args class-body
   class-body =  <'{'> methods <'}'>
   methods = method+
   method = method-name <':'> args <'=>'> method-body
   method-body = expression
   <value> = string | num
   string = \" #'.+' \"
   num = #'[0-9]+'
   <expression> = symbol | method-invocation | constructor | value
   variable = <'let'> symbol <'='> expression
   constructor = class-name invoke-args
   invoke-args = <'('> (expression <','?> expression*)* <')'>
   method-invocation = object <'.'> method-name invoke-args?
   object = symbol | constructor
   method-name = symbol
   args = <'('> (symbol <','?> symbol*)* <')'>
   class-name = (#'[A-Z]' class-name-ending)
   <class-name-ending> = valid-chars
   self = 'self'
   <valid-chars> = !'let' !'self' #'[^0-9\\(\\), \\s\\.:\\\"\\'][^\\(\\), \\s\\.:\\\"\\']*'
   symbol = valid-chars | self"

   :auto-whitespace :standard))

(def trans
  (partial insta/transform {:class-name str}))

(defn parse [string transformer]
  (->> (parser string)
       (transformer)))

(defn standard-parse [string]
  (parse string trans))


(defn javascript-parse [string]
  (->> (parser string)
       (insta/transform
        {
         :class-name str
         :constructor (fn [c args] (str c "(" args ")"))
         :invoke-args (fn [& args] (clojure.string/join "," args))
         :args (fn [& args] (str "(" (clojure.string/join "," args) ")"))
         :num identity
         :class-body (fn [body] (str body "\n return this.self"))
         :class (fn [class-name args body] (str "var " class-name " = function" args "{" body "}"))
         :methods (fn [& meths] (str "this.self = {" (clojure.string/join ",\n" meths) "}"))
         :method (fn [meth-name args body]
                   (str meth-name ": function" args "{ return " body "; }.bind(this)" ))
         :method-body identity
         :method-name identity
         :method-invocation (fn ([o m] (str o "." m))
                                ([o m args] (str o "." m "(" args ")")))
         :object identity
         :self (fn [self] "this.self")
         :variable (fn [n v] (str "var " n " = " v))
         :symbol identity})))

(defn create-js [code]
  (javascript-parse code))


(defn run-code [code]
  (clojure.string/join
   "\n"
   (map js/eval
        (create-js code))))


(defonce app-state
  (atom {:text
            "class Point(x, y) {
  x: () => x
  y: () => y
}
let x = Point(1,2)
x.x()
x.y()"}))

(defn main []
  (om/root
   (fn [app owner]
     (reify
       om/IRender
       (render [_]
               (dom/div nil
                        (dom/span #js {:style #js {:float "left"}}
                          (dom/textarea #js {:onChange (fn [e] (om/update! app [:text] (.. e -target -value)))
                                         :value (:text app)}))
                        (dom/span #js {:style {:float "left"}}
                          (dom/pre nil (clojure.string/join "\n" (create-js (:text app)))))
                        (dom/pre nil (run-code (:text app)))))))
    app-state
    {:target (. js/document (getElementById "app"))}))
