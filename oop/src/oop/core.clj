(ns oop.core
  (:use [oop.macros]
        [oop.essense]))



(obj-> (App (Lam "x" (Add (Var "x") (Var "x")))
            (Add (Con 10) (Con 11)))
       (:interp EmptyEnv)
       :show)
