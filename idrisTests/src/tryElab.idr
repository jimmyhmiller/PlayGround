import Pruviloj
import Pruviloj.Induction
-- Local Variables:
-- idris-load-packages: ("pruviloj")
-- End:


%language ElabReflection


auto : Elab ()
auto =
  do compute
     attack
     try intros
     hs <- map fst <$> getEnv
     for_ hs $
       \ih => try (rewriteWith (Var ih))
     hypothesis <|> search
     solve




partial
mush : Elab ()
mush =
    do attack
       n <- gensym "j"
       intro n
       try intros
       ignore $ induction (Var n) `andThen` auto
       solve


plusAssoc : (j, k, l : Nat) -> plus (plus j k) l = plus j (plus k l)
plusAssoc = %runElab mush


