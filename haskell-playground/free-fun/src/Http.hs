module Http where


data Http = Get String | Post String String

fetch :: Http -> IO ()
fetch (Get url) = putStrLn $ "Get " ++ url
fetch (Post url body) = putStrLn $ "Post " ++ url ++ " with body " ++ body

instructions = [Get "google.com", Post "twitter.com" "Fun with Free Monads"]

main :: IO ()
main = do
  mapM fetch instructions
  return ()
