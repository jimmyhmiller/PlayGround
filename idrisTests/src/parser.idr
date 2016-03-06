
data Parser : Type -> Type where
  Par : (String -> Maybe (a, String)) -> Parser a


instance Functor Parser where
    map m (Par f) = Par (\inp => (case f inp of
                                     Nothing => Nothing
                                     (Just (x, inp')) => Just (m x, inp')))


instance Applicative Parser where
    pure x = Par (\inp => Just (x, inp))
    (<*>) (Par f) (Par g) = Par (\inp => (case f inp of
                                               Nothing => Nothing
                                               (Just (f', inp')) => 
                                                 (case g inp' of
                                                   Nothing => Nothing
                                                   (Just (x, inp'')) => Just (f' x, inp''))))


instance Monad Parser where
    (>>=) (Par g) f = Par (\inp => (case g inp of
                                         Nothing => Nothing
                                         (Just (x, inp')) => (case f x of
                                                                        (Par f') => f' inp')))



instance Alternative Parser where
    empty = Par (\inp => Nothing)
    (<|>) (Par f) (Par g) = Par (\inp => (case f inp of
                                               Nothing => g inp
                                               x => x))


item : Parser Char
item = Par (\inp => (case unpack inp of
                          [] => Nothing
                          (x :: xs) => Just (x, pack xs)))






sat : (Char -> Bool) -> Parser Char                                                                 
sat p = do 
  x <- item
  if p x then pure x else empty
  
char : Char -> Parser Char
char x = sat (== x)

digit : Parser Char
digit = sat isDigit

lower : Parser Char
lower = sat isLower

upper : Parser Char
upper = sat isUpper

twoLower : Parser String
twoLower = do 
  x <- lower
  y <- lower
  pure $ pack [x, y]



runParserPartial : Parser a -> String -> Maybe (a, String)
runParserPartial (Par f) x = f x

runParserFull : Parser a -> String -> Maybe a
runParserFull (Par f) x = case f x of
                               Nothing => Nothing
                               (Just (a, inp)) => (case unpack inp of
                                                        [] => Just a
                                                        xs => Nothing)


