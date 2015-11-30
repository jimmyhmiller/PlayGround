%default total

  

data PO : a -> a -> Type where
  PoRefl : PO a a
  PoBin : PO a b -> PO b c -> PO a c

    

transitive : a = b -> b = c -> a = c   
transitive Refl Refl = Refl

poReflexive : x -> PO x x
poReflexive x = PoRefl

poTransitive : PO a b -> PO b c -> PO a c
poTransitive PoRefl x1 = x1
poTransitive (PoBin x y) PoRefl = PoBin x y
poTransitive (PoBin x y) (PoBin z w) = PoBin x (PoBin y (PoBin z w))


poAntiSymetric : a = b -> PO a b -> PO b a
poAntiSymetric Refl PoRefl = PoRefl
poAntiSymetric Refl (PoBin x y) = poTransitive x y





poAntiSymetric2 : PO a b -> PO b a -> a = b
poAntiSymetric2 PoRefl x1 = Refl
poAntiSymetric2 (PoBin x y) PoRefl = Refl
poAntiSymetric2 {a} {b} (PoBin x y) (PoBin z w) = let p = (pro3 x y z w) in 
  transitive (pro5 p x w) (pro4 p y z)
  where 
    pro3 : (x : PO a b1) -> (y : PO b1 b) -> (z : PO b b2) -> (w : PO b2 a) -> b2 = b1
    pro3 x y z w = poAntiSymetric2 (poTransitive w x) (poTransitive y z)
    
    pro4 : b2 = b1  -> (y : PO b1 b) -> (z : PO b b2) -> b1 = b
    pro4 Refl y z = poAntiSymetric2 y z
    
    pro5 : b2 = b1 -> (x : PO a b1) -> (w : PO b2 a) -> a = b1
    pro5 Refl x w = poAntiSymetric2 x w
    
