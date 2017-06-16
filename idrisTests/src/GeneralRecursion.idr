infixr 5 !!
infixr 5 ??

data General : (s : Type) -> (t : s -> Type) -> (x : Type) -> Type where
  (!!) : x -> General s t x
  (??) : (es : s) -> (t es -> General s t x) -> General s t x
 
fold : {s, x, y : Type} -> {t : s -> Type} -> (x -> y) -> ((es : s) -> (t es -> y) -> y) -> General s t x -> y 
fold r c ((!!) x) = r x
fold r c (s ?? k) = c s (\t => fold r c (k t))

--Functor (General s t) where
--  map func ((!!) x) = (!!) (func x)
--  map func (es ?? f) = ?Functor_rhs_1

--Monad (General s t) where
--  (>>=) x f = ?Monad_rhs_1
--  join x = ?Monad_rhs_2

(>>=) : {s, x, y : Type} -> {t : s -> Type} -> General s t x -> (x -> General s t y) -> General s t y
(>>=) g k = fold k (??) g 


call : {s : Type} -> {t : s -> Type} -> (es : s) -> General s t (t es)
call es = es ?? (!!) 

pig : (s : Type) -> (t : s -> Type) -> Type
pig s t = (es : s) -> General s t (t es)


fusc : pig Nat (\ x => Nat)
fusc Z = (!!) 0
fusc (S n) = call n >>= (\f => call f >>= (\ff => (!!) (S ff)))
