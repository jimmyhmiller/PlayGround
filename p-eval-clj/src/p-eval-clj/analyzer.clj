(ns p-eval-clj.analyzer
  (:gen-class)
  (:require [clojure.tools.analyzer.jvm :as ana]
            [clojure.tools.analyzer.passes.jvm.emit-form :as e]
            [clojure.tools.analyzer.ast :as ast]
            [meander.syntax.zeta :as zyntax]
            [clojure.string :as string]))


(defn logic-variable? [x]
  (and (symbol? x)
       (string/starts-with? (name x) "?")))

(defn parse-with-quotes [pattern]
  (clojure.walk/postwalk
   (fn [x] (if (logic-variable? x)
             `(quote ~x)
             x))
   (zyntax/parse pattern)))



(defn evaluate-one [expr]
  (let [expr-result (ast/prewalk (ana/analyze expr)
                           (fn [ast]
                            
                             (try (ana/analyze (eval (e/emit-form ast)))
                                  (catch Exception e
                                    ast))))]
    
    (e/emit-form expr-result)))





(defn evaluate-many [exprs]
  (mapv evaluate-one exprs))

(evaluate-many
 '[(defn insert [coll k v]
     (if (empty? coll)
       [[k v]]
       (conj coll [k v])))


   (defn lookup [coll x]
     (if (clojure.core/empty? coll)
       nil
       (if (clojure.core/= (clojure.core/ffirst coll) x)
         (clojure.core/second (clojure.core/first coll))
         (lookup (clojure.core/rest coll) x))))


   
   (defn interpret [ast form smap]
     (let [tag (clojure.core/get ast :tag)]

       (cond
         (clojure.core/= tag :literal)
         (let [x (clojure.core/get ast :form)]
           (if (clojure.core/= x form)
             (clojure.core/list smap)))

         (clojure.core/= tag :logic-variable)
         (let [other-form (lookup smap ast)]
           (if other-form
             (if (clojure.core/= other-form form)
               (clojure.core/list smap))
             (clojure.core/list (insert smap ast form))))

         (clojure.core/= tag :cat)
         (if (clojure.core/seq form)
           (let [next-one (clojure.core/next form)]
             (clojure.core/mapcat
              (fn [smap]
                (interpret (clojure.core/get ast :next) next-one smap))
              (interpret (clojure.core/get ast :pattern) (clojure.core/nth form 0) smap))))

         (clojure.core/= tag :vector)
         (if (clojure.core/vector? form)
           (let [pattern (clojure.core/get ast :pattern)]
             (interpret pattern form smap)))

         (clojure.core/= tag :empty)
         (if (clojure.core/seq form)
           nil
           (clojure.core/list smap))

         (clojure.core/= tag :seq)
         (if (seq? form)
           (let [pattern (get ast :pattern)]
             (interpret pattern form smap)))
         
         :else ast)))

   (interpret
    {:tag :vector,
     :pattern
     {:tag :cat,
      :pattern {:tag :logic-variable, :symbol '?x, :form '?x},
      :next {:tag :empty}},
     :form ['?x]}
    arg
    [])
   
   ])




 ('interpret 
     (parse-with-quotes '[?x])
     ~'arg
     [])
(ana/analyze+eval 1)

(-> '(let [a a] a)
    (ana/analyze (assoc (ana/empty-env)
                        :locals '{a {:op    :binding
                                     :name  a
                                     :form  a
                                     :local :let}}))
    (ana/analyze ))

(e/emit-form
 (ana/run-passes (ana/analyze '(let [a 2] (let [y 3] a)))))

(clojure.repl/dir e)
