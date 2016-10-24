%hide (<+>)

interface Group a where
  identity : a
  inverse : a -> a
  (<+>) : a -> a -> a
  

interface Group a => VerifiedGroup a where
  prove_identity : (x : a) -> x <+> Main.identity = x
  prove_identity_r : (x : a) -> Main.identity <+> x = x
  prove_inverse : (x : a) -> x <+> inverse x = Main.identity
  prove_inverse_r : (x : a) -> inverse x <+> x = Main.identity
  prove_assoc : {x : a} -> {y : a} -> {z : a} -> x <+> y <+> z = x <+> (y <+> z)
  


prove_unique_identity : VerifiedGroup a => (e : a) -> ((x : a) -> x <+> e = x) -> e = Main.identity
prove_unique_identity e prf = trans (sym (prove_identity_r e)) (prf identity)

prove_unique_inverse : VerifiedGroup a => (x : a) -> (inv : a) -> x <+> inv = Main.identity -> inv <+> x = Main.identity -> inv = inverse x
prove_unique_inverse x inv prf prf1 = ?test
  where
    t1 : x <+> inv = x <+> inverse x
    t1 = trans prf (sym (prove_inverse x))
    
    t2 : inv <+> (x <+> inv) = inv <+> (x <+> inverse x)
    t2 = cong t1
    
    t3 : inv <+> x <+> inv = inv <+> (x <+> inverse x)
    t3 = trans prove_assoc t2
    
    t4 : inv <+> x <+> inv = inv <+> x <+> inverse x
    t4 = trans t3 (sym prove_assoc)
   
 
 
