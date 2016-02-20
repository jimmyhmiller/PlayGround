

data Free : (f : Type -> Type) -> (a : Type) -> Type where
  Pure : a -> Free f a
  Bind : f (Free f a) -> Free f a
  
  

instance Functor f => Functor (Free f) where
  map m (Pure x) = Pure (m x)
  map m (Bind x) =  ?test
