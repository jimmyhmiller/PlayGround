

class Eq a => Poset a where
  op : a -> a -> Dec (a,a)
  
  
infixr 10 ==<
(==<) : Poset a => a -> a -> Dec (a,a)
x ==< y = op x y



contra : (x,y) -> Void
contra x = contra x

lteToPoset : Nat -> Nat -> Dec (Nat, Nat)
lteToPoset x y with (isLTE x y) | Yes _ = Yes (x,y)
lteToPoset x y with (isLTE x y) | No i = No contra



class Poset a => VerifiedPoset a where
  refl : (x : a) -> x ==< x = Yes (x,x)
  transitive : (x, y, z : a) -> x ==< y = Yes (x,y) -> y ==< z = Yes (y, z) -> x ==< z = Yes (x, z)
  antiSymmetric : (x, y :a) -> x ==< y = Yes (x,y) -> y ==< x = Yes (y, x) -> x = y
  
instance Poset Nat where
  op x y = lteToPoset x y 
  
instance VerifiedPoset Nat where
    refl b = ?VerifiedPoset_rhs_4
    transitive x y z prf prf1 = ?VerifiedPoset_rhs_2
    antiSymmetric x y prf prf1 = ?VerifiedPoset_rhs_3


