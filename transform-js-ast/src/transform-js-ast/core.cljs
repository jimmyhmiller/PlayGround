(ns transform-js-ast.core
  (:require [meander.match.delta :as m :include-macros true]
            [meander.strategy.delta :as r :include-macros true]
            ["@babel/parser" :as babel]
            [clojure.pprint :as pprint]))



(def example 
  "function test () {};
const x = 2;
const y = () => {}; 
const z = function() {}")

(defn parse-js [code]
  (-> (babel/parse code)
      js/JSON.stringify
      js/JSON.parse
      (js->clj :keywordize-keys true)))



(m/search (parse-js example)
  ($ (or
      {:type "FunctionDeclaration"
       :id {:name ?name}
       :loc ?loc}

      {:type "VariableDeclarator"
       :id {:name ?name}
       :loc ?loc
       :init {:type (or "FunctionExpression" "ArrowFunctionExpression")}}))
  {:name ?name
   :loc ?loc})




