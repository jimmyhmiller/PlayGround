using (a : Type, m : (Type -> Type))
  data Parser : (Type -> Type) -> a -> Type where
    Par : (String -> m (a, String)) -> Parser m a


instance Monad monad => Functor (Parser monad) where
    map m (Par f) = Par (\inp => do
      (x, inp') <- f inp
      pure (m x, inp'))


instance Monad monad => Applicative (Parser monad) where
    pure x = Par (\inp => pure (x, inp))
    (<*>) (Par cs1) (Par cs2) = Par (\s => do
      (f, s1) <- cs1 s
      (a, s2) <- cs2 s1
      pure (f a, s2))
 
instance Monad monad => Monad (Parser monad) where
    (>>=) {b = Type} (Par g) f = Par (\inp => do 
      (x, inp') <- g inp
      (case f x of
            (Par f') => f' inp'))
            
instance (Alternative monad, Monad monad) => Alternative (Parser monad) where
    empty {a = Type} = Par (\inp => empty)
    (<|>) (Par f) (Par g) = Par (\inp => f inp <|> g inp)
   
    
using (m : Type -> Type)
  instance Semigroup (m (a, String)) => Semigroup (Parser m a) where
    (<+>) (Par f) (Par g) = Par (\inp => f inp <+> g inp)

using (m : Type -> Type)
  class (Alternative m, Monad m) => Parseable (m : Type -> Type) where

instance (Alternative m, Monad m) => Parseable m where



item : Alternative m => Parser m Char
item = Par (\inp => (case unpack inp of
                          [] => empty
                          (x :: xs) => pure (x, pack xs)))



item' : Parser List Char
item' = Par (\inp => (case unpack inp of
                          [] => []
                          (x :: xs) => [(x, pack xs)]))

sat : Parseable m => (Char -> Bool) -> Parser m Char
sat p = do 
  x <- item
  if p x then pure x else empty
  

char : Parseable m => Char -> Parser m Char
char x = sat (== x)


digit :  Parseable m => Parser m Char
digit = sat isDigit

lower : Parseable m => Parser m Char
lower = sat isLower

upper : Parseable m  => Parser m Char
upper = sat isUpper


runParserPartial : Parseable m => Parser m a -> String -> m (a, String)
runParserPartial (Par f) inp = f inp

runParserPartial' : Parser List a -> String -> List (a, String)
runParserPartial' (Par f) y = f y

runParserFull : Parseable m => Parser m a -> String -> m a
runParserFull (Par f) inp = do 
  (x, inp') <- f inp
  (case unpack inp' of
        [] => pure x
        (x :: xs) => empty)




letter : (Parseable m, Semigroup (Parser m Char)) => Parser m Char
letter = lower <+> upper

word :(Parseable m, Semigroup (Parser m String), Semigroup (Parser m Char)) => Parser m String
word = neWord <+> pure ""
     where
       neWord = do
         x <- letter
         xs <- word
         pure $ strCons x xs

--This doesn't work. Find out why.
many : Parseable m => Parser m a -> Parser m (List a)
many p = [x::xs | x <- p, xs <- many p]


manyDigit : Parser List (List Char)
manyDigit = item' *> Par (\inp => [(['t'], inp)])


q : Parser List (List Char)
q = many digit

q' : Parser List Char
q' = lower

q'' : Parser Maybe (List Char)
q'' = many letter


sat' : Parseable m => (Char -> Bool) -> Parser m Char
sat' p = [x | x <- item, p x]

digit' : Parseable m => Parser m Char
digit' = sat' isDigit
