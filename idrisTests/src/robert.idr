import Data.List

%default total

%hide Vect
%hide fact



data Vect : Type -> Nat -> Type where
  Nil : Vect a Z
  (::) : a -> Vect a n -> Vect a (S n)


data Color = Red | Blue | Green


isFavorite : Color -> Bool
isFavorite Red = False
isFavorite Blue = False
isFavorite Green = True


ifSelector : Bool -> Type -> Type -> Type
ifSelector pred x y = if pred then x else y


iffy : (pred : Bool) -> {a, b : Type} -> a -> b -> ifSelector pred a b
iffy False x y = y
iffy True x y = x


transitive : a = b -> b = c -> a = c
transitive Refl Refl = Refl

symmetric : a = b -> b = a
symmetric Refl = Refl

congruent : (f : t -> x) -> a = b -> f a = f b
congruent f Refl = Refl

plusZero : (n : Nat) -> n + Z = n
plusZero Z = Refl
plusZero (S k) = congruent S (plusZero k)


total
fact : Nat -> Nat
fact Z = (S Z)
fact n@(S m) = n * fact m


fact1 : fact 0 = 1
fact1 = Refl

fact5 : fact 5 = 120
fact5 = Refl


identity : a -> a
identity x = x



  


fmap : (a -> b) -> Vect a n -> Vect b n
fmap f [] = []
fmap f (x :: xs) = f x :: fmap f xs


x : List Int
x = [1,2,3,4,5]


getValue : (x : Int) -> (xs : List Int) -> {auto ok : Elem x xs} -> Int
getValue x (x :: ys) {ok = Here} = x
getValue x (y :: ys) {ok = (There z)} = getValue x ys {ok=z}

y : Int
y = getValue 3 [1,2,3,4,5]


q : Vect Int Z
q = []

q' : Vect Int (S Z)
q' = [0]


