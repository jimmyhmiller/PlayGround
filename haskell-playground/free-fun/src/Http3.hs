module Http3 where
import           Text.Format (format)
import           Text.Printf (printf)

type Url = String
type Body = String

data HttpF next
  = Get Url next
  | Post Url Body next
  | Done

instance Functor HttpF where
  fmap f (Get url next)       = Get url $ f next
  fmap f (Post url body next) = Post url body $ f next
  fmap f Done                 = Done

data Fix f = Fix (f (Fix f))

type Http = Fix HttpF

fetch :: HttpF (IO ()) -> IO ()
fetch (Get url next)  = putStrLn ("Get " ++ url) >> next
fetch (Post u b next) = putStrLn ("Post " ++ u ++ ", " ++ b) >> next
fetch Done            = return ()

toString :: HttpF String -> String
toString (Get url next)       = "Get " ++ url ++ "\\n" ++ next
toString (Post url body next) = "Post " ++ url ++ ", " ++ body ++ "\\n" ++ next
toString Done                 = ""

foldFix :: Functor f => (f a -> a) -> Fix f -> a
foldFix f (Fix t) = f (fmap (foldFix f) t)

instructions :: Http
instructions = Fix $ Get "google.com" $
               Fix $ Post "twitter.com" "Fun with Free Monads" $
               Fix Done

httpInt :: HttpF Int
httpInt = Get "int.com" 123

httpString :: HttpF String
httpString = Get "int.com" "String"


http1 :: Fix HttpF
http1 = Fix $ Get "google.com" $ Fix Done

http2 :: Fix HttpF
http2 = Fix $ Get "twitter.com" $
        Fix $ Post "google.com" "Fun with Free Monads" $
        Fix Done

main :: IO ()
main = foldFix fetch instructions
