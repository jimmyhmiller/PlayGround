import Data.List

data Union : List Type -> Type where
  MemberHere : ty -> Union (ty::ts)
  MemberThere : Union ts -> Union (ty::ts)


member : ty -> {auto e: Elem ty ts} -> Union ts
member x {e=Here} = MemberHere x
member x {e=There later} = MemberThere (member x {e=later})

get : Union ts -> {auto e: Elem ty ts} -> Maybe ty
get (MemberHere x)       {e=Here}    = Just x
get (MemberHere x)       {e=There _} = Nothing
get (MemberThere x)      {e=Here}    = Nothing
get (MemberThere later)  {e=(There l)} = get later {e=l}


data UnionFold : (target: Type) -> (union: Type) -> Type where
  Nil : UnionFold a (Union [])
  (::) : (t -> a) -> UnionFold a (Union ts) -> UnionFold a (Union (t::ts))


foldUnion : (fs: UnionFold a (Union ts)) -> Union ts -> a
foldUnion [] (MemberHere _) impossible
foldUnion [] (MemberThere _) impossible
foldUnion (f :: _) (MemberHere y) = f y
foldUnion (f :: xs) (MemberThere y) = foldUnion xs y


unionLift : (a -> b) -> Union ts -> {auto e: Elem a ts} -> Maybe b
unionLift f x {a} with (the (Maybe a) $ get x)
  unionLift f x {a = a} | Nothing = Nothing
  unionLift f x {a = a} | (Just y) = Just $ f y


x : Union [String, Nat, List String]
x = member "Ahoy!"

y : Nat
y = foldUnion [length, id, sum . map length] x

z : Maybe Nat
z = unionLift S x

