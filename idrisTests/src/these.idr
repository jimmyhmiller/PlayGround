data Those : Type -> Type -> Type where
  This : a -> Those a b
  That : b -> Those a b
  These : a -> b -> Those a b

instance (Semigroup a, Semigroup b) => Semigroup (Those a b) where
    (<+>) (This x) (This y) = This (x <+> y)
    (<+>) (This x) (That y) = These x y
    (<+>) (This x) (These y z) = These (x <+> y) z
    (<+>) (That x) (This y) = These y x
    (<+>) (That x) (That y) = That (x <+> y)
    (<+>) (That x) (These y z) = These y (x <+> z)
    (<+>) (These x z) (This y) = These (x <+> y) z
    (<+>) (These x z) (That y) = These x (y <+> z)
    (<+>) (These x z) (These y w) = These (x <+> y) (z <+> w)



instance Functor (Those a) where
    map f (This x) = This x
    map f (That x) = That (f x)
    map f (These x y) = These x (f y)



instance Monoid a => Applicative (Those a) where
    pure = That
    (<*>) (This x) (This y) = This (x <+> y)
    (<*>) (This x) (That y) = This x
    (<*>) (This x) (These y z) = This (x <+> y)
    (<*>) (That f) (This y) = This y
    (<*>) (That f) (That x) = That (f x)
    (<*>) (That f) (These x y) = That (f y)
    (<*>) (These x f) (This y) = This (x <+> y)
    (<*>) (These x f) (That y) = These x (f y)
    (<*>) (These x f) (These y w) = These (x <+> y) (f w)


instance Monoid a => Monad (Those a) where
    (>>=) (This x) f = This x
    (>>=) (That x) f = f x
    (>>=) (These x y) f = case f y of
                               (This z) => This (x <+> z)
                               (That z) => These x z
                               (These s t) => These (x <+> s) t
