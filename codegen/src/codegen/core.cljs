(ns codegen.core
  (:require ["@jimmyhmiller/estel-estree-builder" :as builder]
            [clojure.walk :as walk]
            [astring :as codegen]
            [clojure.string :as string]
            [instaparse.core :as insta]
            [clojure.pprint :as pprint]
            [codegen.grammar :as grammar]))



(defn parser [grammar]
  (insta/parser grammar :auto-whitespace :standard))




(time

 (insta/parses (parser grammar/grammar) grammar/text))

(defn describe [type]
  (mapv keyword 
        (keys (dissoc (js->clj ((aget builder (name type)) #js {}))
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
           (symbol? expr) (name expr)
           :else expr))
   ast))

(defn convert-ast [ast]
  (js->clj (process-ast ast)))

(defn compile [ast]
  (codegen/generate (process-ast ast)))

(describe :property)
(view-node-types)


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

(defn identifier [name]
  [:identifier {:name  name}])

(defn arrow-function [args body]
  [:arrowFunctionExpression
   {:params args
    :body body}])

(defn return [x]
  [:returnStatement {:argument x}])

(defn block [& args]
  [:blockStatement {:body args}])


(defn literal [expr]
  [:literal {:value expr}])

(defn call [f args]
  [:callExpression {:callee f
                    :args args}])

(defn js-if [test consequent alernate]
  (call (arrow-function []
                        (block [:ifStatement {:test test
                                              :consequent (block (return consequent))
                                              :alternate (block (return alernate))}]))
        []))

(def null [:literal {:value nil}])

(defn array [elements]
  [:arrayExpression {:elements elements}])

(defn program [body]
  [:program {:body body}])


(defn object [expr convert-value]
  [:objectExpression 
   {:properties (map (fn [[k v]] 
                       [:property {:key (identifier (name k))
                                   :value (convert-value v)
                                   :kind "init"}])
                     expr)}])

(defn lisp->ast [expr]
  (cond
    (number? expr) (literal expr)
    (string? expr) (literal expr)
    (boolean? expr) (literal expr)
    (symbol? expr) (identifier expr)
    (keyword? expr) (literal (name expr))
    (nil? expr) null
    (vector? expr) (array (map lisp->ast expr))
    (map? expr) (object expr lisp->ast)
    (= 'if (first expr)) (js-if (lisp->ast (second expr))
                                (lisp->ast (nth expr 2))
                                (lisp->ast (nth expr 3)))
    (= 'def (first expr)) (const (second expr)
                                 (lisp->ast (nth expr 2)))
    (= 'defn (first expr)) (lisp->ast (list 'def (second expr) 
                                            (list 'fn 
                                                  (nth expr 2)
                                                  (nth expr 3))))
    (= 'fn (first expr)) (arrow-function (map lisp->ast (second expr)) 
                                         (lisp->ast (nth expr 2)))
    (seq? expr) (call (lisp->ast (first expr))
                      (map lisp->ast (rest expr)))))


(defn program->ast [exprs]
  (program (map lisp->ast exprs)))

(js/eval)
(compile
 (program->ast 
  '[(defn x [y] {:x 2})]))



(.-nullLiteral builder)

(.toString (.-callExpression builder ))


(compile
 (const "f" (arrow-function 
             '[x y]
             (block 'x (return 'y)))))



(compile example)

(compile example2)

