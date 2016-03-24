import Effects
import Effect.StdIO

data Free : (f : Type -> Type) -> (a : Type) -> Type where
  Pure : a -> Free f a
  Bind : f (Free f a) -> Free f a
  
  

instance Functor f => Functor (Free f) where
  map f (Pure x) = Pure (f x)
  map f (Bind x) = assert_total (Bind (map (map f) x))

instance Functor f => Applicative (Free f) where
  pure = Pure

  (Pure f) <*> x = map f x
  (Bind f) <*> x = assert_total (Bind (map (<*> x) f))

instance Functor f => Monad (Free f) where
  (Pure x) >>= f = f x
  (Bind x) >>= f = assert_total (Bind (map (>>= f) x))


Url : Type
Url = String


data Console a = Put String a | Get (String -> a)




interpret : Free Console a -> Eff a [STDIO]
interpret (Pure x) = return x
interpret (Bind (Put s next)) = do
  putStr s
  interpret next
interpret (Bind (Get f)) = do
  s <- getStr
  interpret (f s)



x : Free Console String
x = Bind (Put "hello" (Bind (Get Pure)))
  

y : Eff String [STDIO]
y = interpret x
