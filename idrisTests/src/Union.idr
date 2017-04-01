
import Data.List

  
data Optional : Type -> Type where
 NotAThing : Optional a
 Something : a -> Optional a

data SimpleUnion : Type -> Type -> Type where
  Left : a -> SimpleUnion a b
  Right : b -> SimpleUnion a b

x : SimpleUnion Int (SimpleUnion String Bool)
x = Left 2


data Union : List Type -> Type where
  MemberHere : ty -> Union (ty::ts)
  MemberThere : Union ts -> Union (ty::ts)


member : ty -> {auto e: Elem ty ts} -> Union ts
member x {e = Here} = MemberHere x
member x {e = There later} =
  MemberThere (member x {e = later})


get : Union ts -> {auto e: Elem ty ts} -> Maybe ty
get (MemberHere x)  {e = Here}    = Just x
get (MemberHere x)  {e = There _} = Nothing
get (MemberThere x) {e = Here}    = Nothing
get (MemberThere later) {e = There l} =
  get later {e=l}


q : Union [String, Int]
q = member (the Int 2)

z : Maybe Int
z = get q


data UnionFold : (target: Type) -> (union: Type) -> Type where
  Nil : UnionFold a (Union [])
  (::) : (t -> a) -> UnionFold a (Union ts) -> UnionFold a (Union (t::ts))


foldUnion : (fs: UnionFold a (Union ts)) -> Union ts -> a
foldUnion [] (MemberHere _) impossible
foldUnion [] (MemberThere _) impossible
foldUnion (f :: _) (MemberHere y) = f y
foldUnion (f :: xs) (MemberThere y) = foldUnion xs y


stuff : Union [String, Nat, List String]
stuff = member (the Nat 23)

f : Union [String, Nat, List String] -> Nat
f x = foldUnion [length, id, sum . map length] x


asdf : UnionFold Int (Union [Int, Nat])
asdf = [(+2), toIntNat]

mapUnion : (a -> b) -> Union ts -> {auto e: Elem a ts} -> Maybe b
mapUnion f x = map f (get x)

number : Nat
number = 3

g : Union [String, Nat, List String] -> Maybe Nat
g x = mapUnion S x

retract : Union xs -> {auto p: Elem ty xs} -> Either (Union (dropElem xs p)) ty
retract (MemberHere x) {p = Here} = Right x
retract (MemberHere x) {p = (There _)} = Left (MemberHere x)
retract (MemberThere x) {p = Here} = Left x
retract (MemberThere x) {p = (There later)} = either (Left . MemberThere) Right $ retract x {p = later}


retracted : Either (Union [Nat, List String]) String
retracted = retract stuff

