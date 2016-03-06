data Continue : Type -> Type -> Type where
  Cont : ((a -> r) -> r) -> Continue r a



instance Functor (Continue r) where
    map m (Cont f) = Cont (\y => f (y . m))
    
    
instance Applicative (Continue r) where
    pure x = Cont (\f => f x)
    (<*>) (Cont f) y = Cont ?Applicative_rhs_1
