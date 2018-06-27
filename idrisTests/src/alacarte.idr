import Data.List

data Fix : (f : Type -> Type) -> Type where
  In : f (Fix f) -> Fix f


infixr 5 :+:

data (:+:) : (f : Type -> Type) -> (g : Type -> Type) -> (a : Type) -> Type where
  L : f a -> (f :+: g) a
  R : g a -> (f :+: g) a


data Value r = Val Int
data Sum r = Add r r

Functor Value where
    map m (Val x) = Val x
    
Functor Sum where
    map m (Add x y) = Add (m x) (m y)


Expr : (Type -> Type) -> Type
Expr = Fix


data Union : List Type -> Type where
  MemberHere : ty -> Union (ty::ts)
  MemberThere : Union ts -> Union (ty::ts)


member : ty -> {auto e: Elem ty ts} -> Union ts
member x {e = Here} = MemberHere x
member x {e = There later} =
  MemberThere (member x {e = later})




data Sub : List a -> List a -> Type where
  SubZ : Sub [] ys
  SubK : Sub xs ys -> Elem ty ys -> Sub (ty::xs) ys


implementation (Functor f, Functor g) => Functor (f :+: g) where
    map m (L x) = L (map m x)
    map m (R x) = R (map m x)


generalize : (u: Union xs) -> {auto s: Sub xs ys} -> Union ys
generalize (MemberHere x) {s = (SubK _ z)} =
  member x {e = z}
generalize (MemberThere x) {s = (SubK y _)} =
  generalize x {s=y}

interface (Functor sub, Functor sup) => (:<:) (sub : Type -> Type) (sup : Type -> Type) where
  inj : sub a -> sup a


infixr 5 :<:

implementation Functor f => (:<:) f f where
    inj x = x

implementation (Functor f, Functor g) => (:<:) f (f :+: g) where 
    inj x = L x
    
    
--implementation (Functor f, Functor g, Functor h, f :<: g) => (:<:) f (h :+: g) where
--    inj x = R (inj x)
--With out overlapping instances I'm not sure how to manage this.

inject : g :<: f => g (Expr f) -> Expr f
inject x = In (inj x)

val : (Value :<: f) => Int -> Expr f
val n = inject (Val n)

add : (Sum :<: f) => Expr f -> Expr f -> Expr f
add x y = inject (Add x y)


--x : Expr (Sum :+: Value)
--x = add (val 300) (val 200)



