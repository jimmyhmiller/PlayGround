(ns testing-stuff.match
  (:require [clojure.core.match :refer [ split-matrix literal-case-matrix-splitter matrix-splitter  non-local-literal-pattern? match analyze emit-matrix occurrences rows action-for-row clj-form compile]]
            [clojure.pprint :as pp])
  (:use [clojure.core.match]
        [clojure.core.match.protocols]))


(clojure.repl/dir clojure.core.match.protocols)

(defmacro build-matrix [vars & clauses]
  `(emit-matrix '~vars '~clauses false))
 
(defmacro m-to-matrix [vars & clauses]
  `(-> (build-matrix ~vars ~@clauses)
       pprint-matrix))






(defn source-pprint [source]
  (binding [pp/*print-pprint-dispatch* pp/code-dispatch
            pp/*print-suppress-namespaces* true]
    (pp/pprint source)))


(defmacro m-to-clj [vars & clauses]
  (binding [clojure.core.match/*line* (-> &form meta :line)
            clojure.core.match/*locals* &env
            clojure.core.match/*warned* (atom false)]
    (try 
      (-> (clj-form vars clauses)
        source-pprint)
      (catch AssertionError e
        `(throw (AssertionError. ~(.getMessage e)))))))

(defn pprint-matrix
  ([pm] (pprint-matrix pm 4))
  ([pm col-width]
     (binding [*out* (pp/get-pretty-writer *out*)]
       (print "|")
       (doseq [o (occurrences pm)]
         (pp/cl-format true "~4D~7,vT" o col-width))
       (print "|")
       (prn)
       (doseq [[i row] (map-indexed (fn [p i] [p i]) (rows pm))]
         (print "|")
         (doseq [p (:ps row)]
           (pp/cl-format true "~4D~7,vT" (str p) col-width))
         (print "|")
         (print " " (action-for-row pm i))
         (prn))
       (println))))





(defmacro m-to-dag [vars & clauses]
  (binding [clojure.core.match/*line* (-> &form meta :line)
            clojure.core.match/*locals* &env
            clojure.core.match/*warned* (atom false)]
    `~(-> (emit-matrix vars clauses)
        compile
        pp/pprint)))


(clojure.repl/dir clojure.core.match)



(match [[1]]
       [(:or [10] [])] :a0
       [(:or [1] [])] :a1
       :else :else)



(def matrix 
  (build-matrix
   [[1]]

   [(:or [10] [])] :a0
   [(:or [1] [])] :a1
   :else :else))

(:v (first (:ps (first (second (rows matrix))))))
(:v (first (:ps (first (first (rows matrix))))))

(groupable?
 (first (:ps (first (second (rows matrix)))))
 (first (:ps (first (first (rows matrix))))))

(specialize-matrix (OrPatterns. [[(:or [10] []) (:or [1] [])]] {}) matrix)


(defn expand-matrix' [matrix col]
  (loop [matrix matrix]
    (let [p (first (column matrix col))]
      (println p)
      (if (pseudo-pattern? p)
        (recur (specialize matrix p))
        matrix))))




(deftype OrPatterns [ps _meta]
  

  Object
  (toString [this]
    (str ps))
  (equals [_ other]
    (and (instance? OrPatterns other) (= ps (:ps other))))

  clojure.lang.IObj
  (meta [_] _meta)
  (withMeta [_ new-meta]
    (OrPatterns. ps new-meta))

  clojure.lang.ILookup
  (valAt [this k]
    (.valAt this k nil))
  (valAt [this k not-found]
    (case k
      :ps ps
      ::tag ::or
      not-found))

  ISpecializeMatrix
  (specialize-matrix [this matrix]
    (let [rows  (rows matrix)
          ocrs  (occurrences matrix)
          nrows (specialize-or-pattern-matrix rows this ps)]
      (println nrows)
      (pattern-matrix nrows ocrs))))

(specialize matrix p)

(def col (choose-column matrix))

matrix



(expand-matrix matrix col)

(pseudo-pattern? (first (column matrix col)))

(let [p (first (column matrix col))
      rows  (rows matrix)
      ocrs  (occurrences matrix)
      nrows (specialize-or-pattern-matrix rows p  [(:or [10] [])])]
  (.toString (column matrix col)))


(macroexpand '(match
           [[1]]
           [(:or [10] [])] :a0
           [(:or [1] [])] :a1
           :else :else))


(defmethod groupable? [::vector ::vector]
  [a b]
  (and (groupable? (:v a) (:v b))
       (= (:rest? a) (:rest? b))
       (= (:size a) (:size b))))

(match
   [[1 2 3]]
   [(:or [_ _ 2] 
         [3 _ _])] :a0

   [(:or [_ _ 1] 
         [1 _ _])] :a1
   :else :else)

(let
 [x [1]]
  (cond (and (vector? x) (== (count x) 1) (= (nth x 0)  3)) :a0
        (and (vector? x) (== (count x) 0)) :a0
        (and (vector? x) (== (count x) 1) (= (nth x 0) 3)) :a1
        (and (vector? x) (== (count x) 0)) :a1
        :else :else))


(let*
 [ocr-26592 [1]]
 (try
  (clojure.core/cond
   (clojure.core/and
    (clojure.core/vector? ocr-26592)
    (clojure.core/== (clojure.core/count ocr-26592) 1))
   (try
    (clojure.core/let
     [ocr-26592_0__26594 (clojure.core/nth ocr-26592 0)]
     (clojure.core/cond
      (clojure.core/= ocr-26592_0__26594 3)
      :a0
      :else
      (throw clojure.core.match/backtrack)))
    (catch
     Exception
     e__25806__auto__
     (if
      (clojure.core/identical?
       e__25806__auto__
       clojure.core.match/backtrack)
      (do
       (try
        (clojure.core/let
         [ocr-26592_0__26594 (clojure.core/nth ocr-26592 0)]
         (clojure.core/cond
          (clojure.core/= ocr-26592_0__26594 3)
          :a0
          :else
          (throw clojure.core.match/backtrack)))
        (catch
         Exception
         e__25806__auto__
         (if
          (clojure.core/identical?
           e__25806__auto__
           clojure.core.match/backtrack)
          (do
           (try
            (clojure.core/let
             [ocr-26592_0__26594 (clojure.core/nth ocr-26592 0)]
             (clojure.core/cond
              (clojure.core/= ocr-26592_0__26594 3)
              :a1
              :else
              (throw clojure.core.match/backtrack)))
            (catch
             Exception
             e__25806__auto__
             (if
              (clojure.core/identical?
               e__25806__auto__
               clojure.core.match/backtrack)
              (do
               (try
                (clojure.core/let
                 [ocr-26592_0__26594 (clojure.core/nth ocr-26592 0)]
                 (clojure.core/cond
                  (clojure.core/= ocr-26592_0__26594 3)
                  :a1
                  :else
                  (throw clojure.core.match/backtrack)))
                (catch
                 Exception
                 e__25806__auto__
                 (if
                  (clojure.core/identical?
                   e__25806__auto__
                   clojure.core.match/backtrack)
                  (do (throw clojure.core.match/backtrack))
                  (throw e__25806__auto__)))))
              (throw e__25806__auto__)))))
          (throw e__25806__auto__)))))
      (throw e__25806__auto__))))
   :else
   (throw clojure.core.match/backtrack))
  (catch
   Exception
   e__25806__auto__
   (if
    (clojure.core/identical?
     e__25806__auto__
     clojure.core.match/backtrack)
    (do :else)
    (throw e__25806__auto__)))))
