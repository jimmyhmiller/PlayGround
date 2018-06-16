data Parser a = Parser (String -> [(a, String)])

instance Functor Parser where
  fmap f (Parser a) = Parser (\s -> _)

main = putStrLn "Hello World"
