%hide (<+>)

interface Group a where
  (<+>) : a -> a -> a
  identity : a
  inverse : a -> a
  

interface Group a => VerifiedGroup a where
  prove_identity : (x : a) -> (x <+> Main.identity) = x
  prove_identity_r : (x : a) -> (Main.identity <+> x) = x
  prove_inverse : (x : a) -> (x <+> Main.inverse x) = Main.identity
  prove_inverse_r : (x : a) -> (inverse x <+> x) = Main.identity
  prove_assoc : {x : a} -> {y : a} -> {z : a} -> x <+> (y <+> z) = x <+> y <+> z


    


inverse_cancel : VerifiedGroup a => {x : a} -> {y : a} -> Main.inverse x <+> x <+> y = y
inverse_cancel {x} {y} = rewrite (prove_inverse_r x) in rewrite (prove_identity_r y) in Refl


unique_identity : VerifiedGroup g => (e : g) -> ((a : g) -> a <+> e = a) -> e = Main.identity
unique_identity e prf = trans (sym (prove_identity_r e)) (prf identity)


unique_inverse : VerifiedGroup g => (inv : g) -> (x : g) -> (x <+> inv) = Main.identity -> inv = inverse x
unique_inverse inv x prf = t4 
  where
    t1 : x <+> inv = x <+> inverse x
    t1 = trans prf (sym (prove_inverse x))
    
    t2 : inverse x <+> (x <+> inv) = inverse x <+> (x <+> inverse x)
    t2 = cong t1
    
    t3 : inverse x <+> x <+> inv = inverse x <+> x <+> inverse x
    t3 = trans (sym prove_assoc) (trans t2 prove_assoc)
    
    t4 : inv = inverse x
    t4 = trans (sym inverse_cancel) (trans t3 inverse_cancel)




cancel : VerifiedGroup g => (a : g) -> (b : g) -> (c : g) -> a <+> b = a <+> c -> b = c
cancel a b c prf = t3
  where
    t1 : inverse a <+> (a <+> b) = inverse a <+> (a <+> c)
    t1 = cong (prf)
    
    t2 : inverse a <+> a <+> b = inverse a <+> a <+> c
    t2 = trans (sym prove_assoc) (trans t1 prove_assoc)
    
    t3 : b = c
    t3 = trans (sym inverse_cancel) (trans t2 inverse_cancel)
    




      
      




    
