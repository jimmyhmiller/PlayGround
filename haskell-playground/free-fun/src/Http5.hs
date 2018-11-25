{-# LANGUAGE RankNTypes       #-}

module Http5 where

data Free f r = Free (f (Free f r)) | Pure r

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
  = Get Url (String -> next)
  | Post Url Body (String -> next)

instance Functor HttpF where
  fmap f (Get url next) = Get url $ fmap f next
  fmap f (Post url body next) = Post url body $ fmap f next

type Http = Free HttpF

fetch :: HttpF a -> IO a
fetch (Get url f) = do
  putStrLn ("Get " ++ url)
  result <- getLine
  putStrLn $ "Result of get: " ++ result
  return $ f result
fetch (Post url body f) = do
  putStrLn ("Post " ++ url ++ ", " ++ body)
  result <- getLine
  putStrLn $ "Result of post: " ++ result
  return $ f result

foldFree :: Monad m => (forall x . f x -> m x) -> Free f a -> m a
foldFree _ (Pure a)  = return a
foldFree f (Free as) = f as >>= foldFree f

liftF :: Functor f => f r -> Free f r
liftF x = Free (fmap Pure x)

get :: Url -> Http String
get url = liftF $ Get url id

post :: Url -> Body -> Http String
post url body = liftF $ Post url body id

instructions :: Http String
instructions = do
   x <- get "google.com"
   if x == "twitter" then
     post "twitter.com" "Fun with Free Monads"
   else
     post "facebook.com" "Fun with Free Monads"

main :: IO ()
main = do
  foldFree fetch instructions
  return ()
