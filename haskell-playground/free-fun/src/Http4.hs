{-# LANGUAGE RankNTypes #-}

module Http4 where

data Free f r
  = Free (f (Free f r))
  | Pure r

instance Functor f => Functor (Free f) where
  fmap f (Pure a) = Pure (f a)
  fmap f (Free x) = Free (fmap (fmap f) x)

instance Functor f => Applicative (Free f) where
  pure = Pure
  Pure a <*> Pure b = Pure $ a b
  Pure a <*> Free mb = Free $ fmap a <$> mb
  Free ma <*> b = Free $ (<*> b) <$> ma

instance (Functor f) => Monad (Free f) where
  return = Pure
  (Free x) >>= f = Free (fmap (>>= f) x)
  (Pure r) >>= f = f r

type Url = String
type Body = String

data HttpF next
  = Get Url next
  | Post Url Body next

instance Functor HttpF where
  fmap f (Get url next)       = Get url $ f next
  fmap f (Post url body next) = Post url body $ f next

type Http = Free HttpF

fetch :: HttpF a -> IO a
fetch (Get url next) = do
  putStrLn ("Get " ++ url)
  return next
fetch (Post url body next) = do
  putStrLn ("Post " ++ url ++ ", " ++ body)
  return next

foldFree :: Monad m => (forall x . f x -> m x) -> Free f a -> m a
foldFree _ (Pure a)  = return a
foldFree f (Free as) = f as >>= foldFree f

liftF :: Functor f => f r -> Free f r
liftF x = Free (fmap Pure x)

get :: Url -> Http ()
get url = liftF $ Get url ()

post :: Url -> Body -> Http ()
post url body = liftF $ Post url body ()

instructions :: Http ()
instructions = do
   get "google.com"
   post "twitter.com" "Fun with Free Monads"

main :: IO ()
main = foldFree fetch instructions
