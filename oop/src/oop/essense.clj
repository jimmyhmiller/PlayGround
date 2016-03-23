(ns oop.essense
  (:use [oop.macros]))



(defclass Unit [a]
  {:bind (fn [self k] (k a))
   :show (fn [self] (str "Success: " (a :showval)))})

(defclass ErrorM [s]
  {:bind (fn [self k] self)
   :show (fn [self] (str "Error " s))})


(defclass Num [i]
  {:showval (fn [self] i)
   :add (fn [self b] (Unit (b :bind
                              (fn [b'] (Num (+ i b'))))))
   :bind (fn [self k] (k i))})

(defclass Fun [f]
  {:showval (fn [self] "<function>")
   :apply (fn [self a] (f a))})

(defclass Lam [x v]
  {:interp (fn [self e] (Unit
                         (Fun
                          (fn [a]
                            (v :interp (e :add x a))))))})


(defclass Var [x]
  {:interp (fn [self e] (e :lookup x))})

(defclass Con [i]
  {:interp (fn [self e] (Unit (Num i)))})

(defclass Add [u v]
  {:interp (fn [self e] (obj-> u
                               (:interp e)
                               (:bind (fn [a]
                                        (obj-> v
                                               (:interp e)
                                               (:bind (fn [b] (a :add b))))))))})
(defclass Wrong
  {:showval (fn [self] "<wrong>")})



(defclass App [t u]
  {:interp (fn [self e] (obj-> t
                               (:interp e)
                               (:bind (fn [f]
                                        (obj-> u
                                               (:interp e)
                                               (:bind (fn [a] (f :apply a))))))))})

(defclass AddEnv [e y b]
  {:lookup (fn [self x] (if (= x y)
                          (Unit b)
                          (e :lookup x)))
   :add (fn [self x a] (AddEnv self x a))})

(defclass EmptyEnv
  {:lookup (fn [self x] (ErrorM (str "unbound variable " x)))
   :add (fn [self x a] (AddEnv self x a))})



