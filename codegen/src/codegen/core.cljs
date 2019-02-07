(ns codegen.core
  (:require ["@jimmyhmiller/estel-estree-builder" :as builder]
            [clojure.walk :as walk]
            [astring :as codegen]))


(defn describe [type]
  (mapv keyword (keys (dissoc (js->clj ((aget builder (name type)) #js {}))
                             "type"
                             "loc"))))

(defn view-node-types []
  (sort
   (map (juxt identity describe)
        (map keyword
             (js->clj
              (.keys js/Object builder))))))

(defn process-ast [ast]
  (walk/postwalk
   (fn [expr]
     (cond (and (vector? expr)
                (not (map-entry? expr))
                (keyword? (first expr)))
           (let [node-type (name (first expr))
                 node-builder (aget builder node-type)]
             (if-not node-builder
               (throw (ex-info (str "No node type found " node-type) {:node-type node-type
                                                                      :expr expr}))
               (node-builder (second expr))))
           (map? expr) (clj->js expr)
           (symbol? expr) (.identifier builder #js {:name (name expr)})
           :else expr))
   ast))

(defn convert-ast [ast]
  (js->clj (process-ast ast)))

(defn compile [ast]
  (codegen/generate (process-ast ast)))

(describe :variableDeclaration)
(view-node-types)

(defn equal-or-var? [x y]
  (or (symbol? y) (= x y)))

(defn pattern-matches [values pattern]
  (if (not= (count values) (count pattern))
    false
    (every? (partial apply equal-or-var?) (map vector values pattern))))


(pattern-matches [1] [1])
(pattern-matches [2] [2])
(pattern-matches [2] '[n])

(def example
  [:functionDeclaration
   {:id [:identifier {:name "helloWorld"}]
    :params [[:identifier {:name "name"}]]
    :body [:functionBody
           {:body
            [[:returnStatement
              {:argument [:binaryExpression
                          {:operator "+"
                           :left [:literal {:value "Hello"}]
                           :right [:identifier {:name "name"}]}]}]]}]}])

(def example2
  [:program
   {:body
    [[:variableDeclaration
      {:declarations
       [[:variableDeclarator
         {:id [:identifier {:name "f"}]
          :init [:arrowFunctionExpression
                 {:params [[:identifier {:name "x"}]]
                  :body [:identifier {:name "x"}]}]}]]
       :kind "const"}]]}])


(defn const [name value]
  [:variableDeclaration
   {:declarations
    [[:variableDeclarator
      {:id [:identifier {:name name}]
       :init value}]]
    :kind "const"}])

(defn arrow-function [args body]
  [:arrowFunctionExpression
   {:params (mapv symbol args)
    :body body}])

(defn return [x]
  [:returnStatement {:argument x}])

(defn block [& args]
  [:blockStatement {:body args}])

(compile
 (const "f" (arrow-function 
             '[x y]
             (block 'x (return 'y)))))



(compile example)

(compile example2)

