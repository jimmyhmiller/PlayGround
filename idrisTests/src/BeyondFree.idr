

Free : Type
Free = Type -> (Type -> Type) -> (Type -> Type) -> Type -> Type

data PureF : Free where
  Pure : a -> PureF e z f a

data EffectF : Free where
  Effect : (f a) -> EffectF e z f a

data FixedF : (t : Free) -> (e : Type) -> (f : Type -> Type) -> (a : Type) -> Type where
  Fixed : (t e (FixedF t e f) f a) -> FixedF t e f a
 
data EitherF : (t1 : Free) -> (t2 : Free) -> Free where
  LeftF : t1 e f z a -> EitherF t1 t2 e f z a
  RightR : t2 e f z a -> EitherF t1 t2 e f z a
