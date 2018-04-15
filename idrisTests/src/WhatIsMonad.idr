
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



myError : Throw String Int
myError = do
  x <- Success 1 
  y <- Success 2
  z <- Success 3 
  pure (x + y + z)


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
  (<|>) x y = x



data MyList a = Nil | (::) a (MyList a)

Functor MyList where
  map f [] = []
  map f (x :: xs) = f x :: map f xs

Applicative MyList where
  pure x = [x]
  (<*>) [] y = []
  (<*>) (x :: z) [] = []
  (<*>) (f :: fs) (x :: xs) = (f x) :: (fs <*> xs)

concat : MyList a -> MyList a -> MyList a
concat [] y = y
concat (x :: xs) ys = x :: concat xs ys

Monad MyList where
  (>>=) [] f = []
  (>>=) (x :: xs) f = concat (f x) (xs >>= f)


myList : MyList (Int, Int)
myList = do
  x <- [1,2,3]
  y <- [3,4,5]
  pure (x,y)




data Parser a = Parse (String -> List (a, String))

Functor Parser where
  map func (Parse f) = Parse (\s => do
    (a, s) <- (f s)
    pure (func a, s)
  )

Applicative Parser where
  pure x = Parse (\s => [(x, s)])
  (<*>) (Parse f) (Parse g) = Parse (\s => do
    (f, s) <- f s
    (x, s) <- g s
    pure (f x, s)
  )

Monad Parser where
  (>>=) (Parse g) f = Parse (\s => do
    (a, s) <- g s
    let (Parse g) = (f a)
    g s
  )
  
  
Alternative Parser where
  empty = Parse (\s => [])
  (<|>) (Parse f) (Parse g) = Parse (\s => (case f s of
                                                   [] => g s
                                                   res => res))


item : Parser Char
item = Parse (\s => (case unpack s of
                          [] => empty
                          (x :: xs) => [(x, pack xs)]))


satisfies : (Char -> Bool) -> Parser Char
satisfies p = do
  x <- item
  if p x then pure x else empty



data Reader e a = Read (e -> a)

Functor (Reader e) where
  map f (Read g) = Read (f . g)

Applicative (Reader e) where
  pure x = Read (\e => x)
  (<*>) (Read f) (Read g) = Read (\e => f e (g e))


Monad (Reader e) where
  join (Read f) = Read (\e => (case f e of
                                    (Read f) => f e))


ask : Reader e e
ask = Read (\e => e)


run : e -> Reader e a -> a
run e (Read f) = f e

Url : Type
Url = String

urlSegments : Reader String (List String)
urlSegments = do
  baseUrl <- ask
  pure $ split (== '/') baseUrl


record Config a where
  constructor MkConfig
  baseUrl : String
  send : String -> a
  
 
sendMessage : String -> Reader (Config a) a
sendMessage s = do
  (MkConfig _ sender) <- ask
  pure $ sender s

prodConfig : Config (IO ())
prodConfig = MkConfig "prod.com" putStr

devConfig : Config String
devConfig = MkConfig "dev.com" id

