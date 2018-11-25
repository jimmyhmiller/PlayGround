module Http1 where

import           Data.Foldable (for_)

type Url = String
type Body = String

data Http = Get Url | Post Url Body

fetch :: Http -> IO ()
fetch (Get url)       = putStrLn $ "Get " ++ url
fetch (Post url body) = putStrLn $ "Post " ++ url ++ " with body " ++ body

requests = [Get "google.com", Post "twitter.com" "Fun with Free Monads"]

main :: IO ()
main =
  for_ requests $ \request ->
    fetch request
