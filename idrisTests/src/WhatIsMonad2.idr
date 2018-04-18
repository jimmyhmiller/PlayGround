data Optional a = None | Some a


Functor Optional where
  map func None = None
  map func (Some x) = Some (func x)

Applicative Optional where
  pure x = Some x
  (<*>) None y = None
  (<*>) (Some x) None = None
  (<*>) (Some f) (Some x) = Some (f x)

Monad Optional where
  (>>=) None f = None
  (>>=) (Some x) f = f x
  
Alternative Optional where
  empty = None
  (<|>) None y = y
  (<|>) x None = x
  (<|>) x y = x

x : Optional (Int, Int, Int)
x = do
  a <- Some 2
  b <- None <|> Some 4
  c <- Some 4
  pure (a,b,c)
  
  
  
data Throw e a = Error e | Success a
  
Functor (Throw e) where
  map func (Error x) = Error x
  map func (Success x) = Success (func x)
  
  
Applicative (Throw e) where
  pure x = Success x
  (<*>) (Error x) y = Error x
  (<*>) (Success x) (Error y) = Error y
  (<*>) (Success f) (Success x) = Success (f x)


Monad (Throw e) where 
  join (Error x) = Error x
  join (Success x) = x
  
  
  
y : Throw String (Int, Int, Int)
y = do
  a <- Success 2
  b <- Error "ERROR!!!!"
  c <- Error "Error 2"
  pure (a,b,c)
