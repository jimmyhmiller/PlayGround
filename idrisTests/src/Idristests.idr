module Idristests

import Prelude.Strings
import Data.SortedMap
import Control.Category
import Control.Arrow


data St : Type -> Type -> Type where
  S : (s -> (a, s)) -> St s a


mapState : (a -> b) -> (s -> (a, s)) -> (s -> (b, s))
mapState f g x = case g x of
                      (y, z) => (f y, z)

instance Functor (St s) where
    map m (S f) = S (mapState m f)
    
pureState : a -> (s -> (a, s))
pureState x y = (x, y)

apState : (s -> (a -> b, s)) -> (s -> (a,s)) -> (s -> (b, s))
apState f g s = case f s of
                     (h, y) => (case g y of
                                     (z, s') => (h z, s'))

instance Applicative (St s) where
    pure x = S (pureState x)
    (<*>) (S f) (S g) = S (apState f g)


bindState : (s -> (a, s)) -> (a -> St s b) -> (s -> (b, s))
bindState f g s = case f s of
                       (a, s') => (case g a of
                                        (S h) => h s')

instance Monad (St s) where
    (>>=) (S g) f = S (bindState g f)




fresh : St Int Int
fresh = S inc' where
  inc' : Int -> (Int, Int)
  inc' n = (n, n + 1)


get : St s s
get  = S (get') where
  get' : (s -> (s,s))
  get' x = (x, x)


put : s -> St s ()
put y = S (put') where 
  put' : s -> ((), s)
  put' _ = ((), y)



freshN : St Int Int
freshN = do
  n <- get
  put (n+1)
  return n


n : St Int (Int, Int, Int)
n = do
  a <- fresh
  b <- fresh
  c <- fresh
  return (a, b, c)




runState : s -> St s a -> a
runState x (S f) = case f x of
                        (a, s) => a


data Validation : Type -> Type -> Type  where
  Expect : (a -> Maybe b) -> Validation a b
  
 
mapMaybe : (a1 -> b) -> (a -> Maybe a1) -> (a -> Maybe b)  
mapMaybe f g x = case g x of
                      Nothing => Nothing
                      (Just y) => Just (f y)


instance Functor (Validation a) where
    map f (Expect g) = Expect (f `mapMaybe` g)
   
 

appMaybe : (a -> Maybe (a1 -> b)) -> (a -> Maybe a1) -> (a -> Maybe b)
appMaybe f g x = case g x of
                      Nothing => Nothing
                      (Just y) => (case f x of
                                        Nothing => Nothing
                                        (Just z) => Just (z y))

instance Applicative (Validation a) where
    pure x = Expect (constant) where
      constant : a -> Maybe a1
      constant y = Just x
    (<*>) (Expect f) (Expect g) = Expect (f `appMaybe` g)
    

bindMaybe : (a -> Maybe a1) -> (a1 -> Validation a b) -> (a -> Maybe b)
bindMaybe f g x = case f x of
                       Nothing => Nothing
                       (Just y) => (case g y of
                                         (Expect h) => h x)

    
instance Monad (Validation a) where
    (>>=) (Expect f) g = Expect (f `bindMaybe` g)

nextMaybe : (a -> Maybe b) -> (b -> Maybe c) -> (a -> Maybe c)
nextMaybe f g x = case f x of
                          Nothing => Nothing
                          (Just y) => g y



orMaybe : (a -> Maybe b) -> (a -> Maybe b) -> (a -> Maybe b)
orMaybe f g x = case f x of
                     Nothing => g x
                     y => y
   
instance Alternative (Validation a) where
    empty = Expect (nothing) where
      nothing : b -> Maybe a1
      nothing x = Nothing
    (<|>) (Expect f) (Expect g) = Expect (f `orMaybe` g)


next : Validation a b -> Validation b c -> Validation a c
next (Expect f) (Expect g) = Expect (f `nextMaybe` g)

instance Category Validation where
    id = Expect Just
    (.) x y = y `next` x



liftMaybe : (a -> b) -> (a -> Maybe b)
liftMaybe f x = Just (f x)

firstMaybe : (a -> Maybe b) -> ((a,c) -> Maybe (b, c))
firstMaybe f (a,c) = case f a of
                          Nothing => Nothing
                          (Just x) => Just (x, c)
                          

instance Arrow Validation where
    arrow f = Expect (liftMaybe f)
    first (Expect f) = Expect (firstMaybe f)


arrowLeft : (a -> Maybe b) -> (Either a c -> Maybe (Either b c))
arrowLeft f (Left l) = case f l of
                            Nothing => Nothing
                            (Just x) => Just (Left x)
arrowLeft f (Right r) = Just (Right r)


arrowRight : (a -> Maybe b) -> (Either c a -> Maybe (Either c b))
arrowRight f (Left l) = Just (Left l)
arrowRight f (Right r) = case f r of
                              Nothing => Nothing
                              (Just x) => Just (Right x)



andMaybe : (a -> Maybe b) -> (a -> Maybe b) -> (a -> Maybe (b,b))
andMaybe f g x = case f x of
                      Nothing => Nothing
                      (Just y) => (case g x of
                                        Nothing => Nothing
                                        (Just z) => Just (y, z))



and : Validation a b -> Validation a b -> Validation a (b,b)
and (Expect f) (Expect g) = Expect (f `andMaybe` g)




or : Validation a b -> Validation a b -> Validation a b
or (Expect f) (Expect g) = Expect (f `orMaybe` g)




hasX : Eq a => a -> Validation (List a) a
hasX n = Expect hasN' where
  hasN' : List a -> Maybe a
  hasN' [] = Nothing
  hasN' (x :: xs) = case x == n of
                         False => hasN' xs
                         True => Just n


wordContainsLetter : Char -> String -> Bool
wordContainsLetter letter word = case unpack word of
                              [] => False
                              (x :: xs) => (case letter == x of
                                                 False => wordContainsLetter letter (pack xs)
                                                 True => True)

hasWordWithLetter : String -> Validation String String
hasWordWithLetter letter' = Expect (hasWord') where
  letter : Char
  letter = strHead letter'
  hasWord' : String -> Maybe String
  hasWord' x = case words x of
                    [] => Nothing
                    (x :: xs) => (case wordContainsLetter letter x of
                                       False => hasWord' (unwords xs)
                                       True => Just x)                                  



hasKey : a ->  Validation (SortedMap a b) b
hasKey k = Expect (SortedMap.lookup k)


q : Validation (List Int) Int
q = map id (hasX 2)


hasTwo : Validation (List Int) Int
hasTwo = hasX 2

isOdd : Validation Int Bool
isOdd = Expect isOdd' where
  isOdd' : Int -> Maybe Bool
  isOdd' x = Just (not ((mod x 2) == 0))
  

z : Validation (SortedMap String String) String
z = hasKey "name" >>= hasKey


l : Validation (SortedMap String String) String
l = do
  name <- hasKey "name" <|> hasKey "otherName"
  age <- hasWordWithLetter "3" . hasKey "age"
  return (name ++ age)
  

eval : (Validation a b) -> a -> Maybe b
eval (Expect f) y = f y


x : List Int
x = [1,2,3,4]



main : IO ()
main = putStrLn $ show $ eval ((hasWordWithLetter "z" . hasKey "name") `or` hasKey "age") $ 
  fromList [("name", "jimmy"), ("age", "30")]
