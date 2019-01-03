(ns testing-stuff.erorr-handling
  (:require [sc.api]
            [clojure.walk :as walk]
            [clojure.string :as string]))




(def x 2)
(sc.api/letsc 2
              [x y z])


(defn log [x]
  (println x)
  x)

(defn wrap-expr [context expr]
  `(try
     (sc.api/letsc ~context
                   ~expr)
     nil
     (catch Exception e# 
       (do
         {:exeception e#
          :expr (quote ~expr)}))))

(def ^:dynamic *binding-context*)


(defn find-code-with-errors [context expr]
  (->> expr
       (tree-seq seq? identity)
       (map (partial wrap-expr context))
       (map eval)
       (filter identity)
       (map :expr)
       last))


(sc.api/spy {:sc/spy-ep-post-eval-logger 
                    (fn [arg]
                      (println arg))}
                   (+ 2 2))

(defmacro find-error-expr [expr name]
  `(let [p# (promise)]
     (try
       (sc.api/spy {:sc/spy-ep-pre-eval-logger identity
                    :sc/spy-ep-post-eval-logger 
                    (fn [{e-id# :sc.ep/id 
                          error# :sc.ep/error
                          value# :sc.ep/value
                          {cs-id# :sc.cs/id 
                           expr# :sc.cs/expr} :sc.ep/code-site}]
                      (deliver p#
                               {:error (find-code-with-errors [e-id# cs-id#] expr#)
                                :value value#
                                :cause (when error# (:cause (Throwable->map error#)))
                                :meta-info (meta (find-code-with-errors [e-id# cs-id#] expr#))
                                :function (meta (var ~name))}))}
                   ~expr)
       (catch Exception e#))
     (deref p#)))





(defmacro defn-debug [name args & body]
  `(defn ~name ~args
    (find-error-expr ~@body ~name)))




(defn-debug complicated-math [x y z]
  (+ 2 3 (/ x 2) (/ x y (/ 2 z) 
                    (/ z x) (+ x y) (+ 2 (* x y)))))

(complicated-math 1 3 0)

(defn format-line [[line text]]
  (if line
    (str line "| " text)
    (str "   " text)))

(defn insert-at [n val vec]
  (let [[before after] (split-at n vec)]
    (concat before [val] after)))



(defn point-to-form [column length]
  [nil (string/join (concat (repeat column " ") (repeat length "^")))])

(point-to-form 2 2)


(defn code-frame [text line column length]
  (let [lines 
        (into [] (-> text
                      (string/split #"\n")))
        numbered-lines
        (->> (range (- line 1) (+ line 4))
             (map (fn [i] [i (get lines i)]))
             (into []))
        offset (- line (count numbered-lines))]
    (str
     (->> numbered-lines
              (insert-at 4 (point-to-form column length))
              
              (map format-line)
              (string/join "\n")))))

(def current-file
  (slurp "/Users/jimmy/Documents/Code/PlayGround/testing-stuff/src/testing_stuff/erorr_handling.clj"))

(str "We found the following code caused a \"Divide by zero\" error"
     "\n"
     "\n"
     (code-frame 
      current-file 77 21 (count "(/ 2 z)")))

