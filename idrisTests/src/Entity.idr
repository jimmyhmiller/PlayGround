


data Fix : (Type -> Type) -> Type where
  MkFix : (f (Fix f)) -> Fix f

data Cofree : (Type -> Type) -> Type -> Type where
  MkCoFree : a -> (f (Cofree f a)) -> Cofree f a

data NewList a r = Empty | Cons a r




MyList : Type -> Type
MyList a = Fix (NewList a)


Functor (NewList a) where
  map func Empty = Empty
  map func (Cons x y) = (Cons x (func y))


unfix : Functor f => (f1 : f a -> a) -> Fix f -> f (Fix f)
unfix f1 (MkFix x) = x

cata : Functor f => (f a -> a) -> (Fix f -> a)
cata f = f . map (cata f) . (unfix f)


length : MyList a -> Int
length = cata (\x => 
  case x of
    Empty => 0
    (Cons _ n) => n + 1)
    

sum : MyList Int -> Int
sum = cata (\x =>
  case x of
        Empty => 0
        (Cons a b) => a + b)

record ProfF a where
  constructor MkProfF
  name : String
  year : Int
  students : List a




Prof : Type
Prof = Fix ProfF

IdProf : Type
IdProf = Cofree ProfF Int



 
