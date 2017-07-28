(ns language.core
  (:require [instaparse.core :as insta]))


(defmacro obj->
  [x & forms]
  (loop [x x, forms forms]
    (if forms
      (let [form (first forms)
            threaded (if (seq? form)
                       (with-meta `(~x ~(first form) ~@(next form)) (meta form))
                       (list x form))]
        (recur threaded (next forms)))
      x)))


(defn dispatcher [obj message & args]
  (cond
   (= message :methods)
   (keys obj)
   (= message :extend)
   (partial dispatcher (merge obj (first args)))
   (clojure.test/function? obj)
   (apply obj (cons message args))
   :else
   (apply (obj message)
          (cons (partial dispatcher obj) args))))


(defmacro defclass
  ([fn-name body]
   `(def ~fn-name
      (partial dispatcher ~body)))
  ([fn-name params body]
   `(defn ~fn-name ~params
      (partial dispatcher ~body))))


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
   <expression> = symbol | method-invocation | constructor | value | self
   variable = <'let'> symbol <'='> expression
   constructor = class-name invoke-args
   invoke-args = <'('> (expression <','?> expression*)* <')'>
   method-invocation = object <'.'> method-name invoke-args?
   object = symbol | constructor | self
   method-name = symbol
   args = <'('> (symbol <','?> symbol*)* <')'>
   class-name = (#'[A-Z]' class-name-ending)
   <class-name-ending> = valid-chars
   self = 'self'
   <valid-chars> = !'let' !'self' #'[^0-9\\(\\), \\s\\.:\\\"\\'][^\\(\\), \\s\\.:\\\"\\']*'
   symbol = valid-chars"
   :auto-whitespace :standard))

(def trans
  (partial insta/transform {:class-name str}))

(defn parse [string transformer]
  (->> (parser string)
       (transformer)))

(defn standard-parse [string]
  (parse string trans))


(defn clojure-parse [string]
  (->> (parser string)
       (insta/transform
        {:class (fn [class-name args body] `(defclass ~class-name ~args ~body))
         :class-name (comp symbol str)
         :class-body identity
         :args (fn [& args] (into [] (map symbol args)))
         :methods merge
         :method-name keyword
         :method (fn [meth-name args body]
                   (let [args' (into [(symbol "self")] args)]
                     {meth-name `(fn ~args' ~body)}))
         :method-body identity
         :object symbol
         :symbol symbol
         :self symbol
         :method-invocation (fn [object method] `(~object ~method))
         :constructor (fn [c args] `(apply ~c '~args))
         :invoke-args list
         :num read-string
         :variable (fn [n v] `(def ~n ~v))})))



(defn javascript-parse [string]
  (->> (parser string)
       (insta/transform
        {:class-name str
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



(defn print-js [js] 
  (doseq [e js]
    (println e)))


(print-js
 (javascript-parse
  "class Point(x, y) {
                    x: () => x
                    y: () => y
                }
                let x = Point(1,2)
                x.x()"))



(comment

(defn generate-mixfix [mixfix]
  (str mixfix " = "
    (clojure.string/join
     " expression "
     (map #(str "<'" % "'>") (clojure.string/split mixfix #"_")))))

(generate-mixfix "if_then_else")

(map eval
(clojure-parse "class Point(x, y) {
                    x: () => x
                    y: () => y
                }
                let x = Point(1,2)
                x.x"))

((Point 1 2) :x)





(standard-parse "class Point(x, y) {
                    x: () => x.y
                    y: () => y
                }")

(standard-parse "Point(x,y)")

(standard-parse "let x = Point(1,2)
                x")

(standard-parse "class Point-Colored(x, y, color) {
                    x: () => x
                    y: () => y
                    color: () => color
                }")





(standard-parse "class Empty() {
    empty: () => True
    contains: (i) => False
    insert: (i) => Insert(self, i)
    union: (s) => s
}")

(standard-parse "class Insert(s, n) {
    empty: () => False
    insert: (i) => Insert(self, i)
    union: (s) => Union(self, s)
}")

(standard-parse "class Union(s1, s2) {
    empty: () => s.empty()
    insert: (i) => Insert(self, i)
    union: (s) => Union(self, s)
}"))
