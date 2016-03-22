%default total

charsToType : (xs : List Char) -> {auto ok: isCons xs = True} -> Type
charsToType ('I' :: 'n' :: 't' :: []) {ok = Refl} = Integer
charsToType ('S' :: 't' :: 'r' :: 'i' :: 'n' :: 'g' :: []) {ok = Refl} = String
charsToType ('I' :: 'n' :: 't' :: x :: xs) {ok = Refl} = Integer -> charsToType $ x :: xs
charsToType ('S' :: 't' :: 'r' :: 'i' :: 'n' :: 'g' :: x :: xs) {ok = Refl} = String -> charsToType $ x :: xs
charsToType (x :: y :: xs) {ok = Refl} = charsToType $ y :: xs
charsToType xs = Unit




isNonEmpty : String -> Bool
isNonEmpty x = isCons (unpack x)

type : (s : String) -> {auto ok : isNonEmpty s = True} -> Type
type s = charsToType (unpack s)
 

double : type "double(Int) : Int"
double x = x * 2


infixr 10 :::

(:::) : Type -> Type -> Type
(:::) x y = (x,y)

infixl 0 ->>

(->>) : Type -> Type -> Type
(->>) x y = x -> y

x : Int ::: Int ::: Int ->> Int
x (a,b,c) = a + b + c
