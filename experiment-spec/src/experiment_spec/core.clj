(ns experiment-spec.core
  (:require [clojure.spec.alpha :as s]))


(s/def ::lang (s/+ ::expr))

(s/def ::expr (s/or :constant ::constant :s-expr ::s-expr))

(s/def ::constant (s/or :int int? :string string?))
(s/def ::s-expr (s/cat :verb (s/or :built-in ::built-in :sym symbol?) 
                       :args (s/* ::expr)))
(s/def ::built-in #{'+ '- '* '/})

(s/conform ::lang '((+ (+ 2 2) 2)))
(s/exercise ::lang)

(s/fdef run
        :args (s/cat :expr ::lang)
        :ret any?)

(defn run [& expr]
  (last (map interpret (s/conform ::lang expr))))

(run "test")

(defmulti interpret first)

(defmethod interpret :constant [[_ [_ val]]]
  val)

(defmethod interpret :s-expr [[_ {:keys [verb args]}]]
  (if (= (first verb) :built-in)
    (apply (resolve (second verb)) (map interpret args))
    (list (second verb) (map interpret args))))

(defmethod interpret :default [x] x)

(resolve '+)

(def x +)

(apply x (list 2 2))

(s/conform ::lang '((+ (+ 2 2) 2)))

(run '(* 2 (+ (+ 2 2) 2)))
(apply + (list (apply + '(2 2)) 2))

(apply + '(2 2))


(s/exercise-fn `run)


