module Http2 where

type Url = String
type Body = String

data Http
  = Get Url Http
  | Post Url Body Http
  | Done

fetch :: Http -> IO ()
fetch (Get url _)       = putStrLn $ "Get " ++ url
fetch (Post url body _) = putStrLn $ "Post " ++ url ++ " with body " ++ body
fetch Done              = return ()

instructions :: Http
instructions =
  Get "google.com" (
    Post "twitter.com" "Fun with Free Monads" Done
  )

run :: (Http -> IO ()) -> Http -> IO ()
run f h@(Get _ next)    = f h >> run f next
run f h@(Post _ _ next) = f h >> run f next
run f Done              = return ()

main :: IO ()
main = run fetch instructions
