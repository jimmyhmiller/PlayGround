
data Optional : Type -> Type where
  Nothing : Optional a
  Some: a -> Optional a



optionalMap : (a -> b) -> Optional a -> Optional b
optionalMap f Nothing = Nothing
optionalMap f (Some x) = Some (f x)

instance Functor Optional where
    map m x = optionalMap m x
    

optionalPure : a -> Optional a
optionalPure x = Some x


optionalApp : Optional (a -> b) -> Optional a -> Optional b
optionalApp Nothing Nothing = Nothing
optionalApp Nothing (Some x) = Nothing
optionalApp (Some x) Nothing = Nothing
optionalApp (Some f) (Some x) = Some (f x)

instance Applicative Optional where
    pure x = optionalPure x
    (<*>) x y = optionalApp x y
    
optionalBind : Optional a -> (a -> Optional b) -> Optional b
optionalBind Nothing f = Nothing
optionalBind (Some x) f = f x


instance Monad Optional where
    (>>=) x f = optionalBind x f

instance Alternative Optional where 
    empty = Nothing
    (<|>) Nothing Nothing = Nothing
    (<|>) Nothing (Some x) = Some x
    (<|>) (Some x) Nothing = Some x
    (<|>) (Some x) (Some y) = Some y



x : Optional Int
x = do 
  a <- Some 3 <|> Nothing
  b <- Some 4
  c <- Nothing <|> Some 1 
  return (a + b + c)
