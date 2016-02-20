import Data.SortedMap


data Validation : Type -> Type -> Type -> Type where
  Expect : (a -> Either b c) -> Validation b a c
  




mapExpect : (a1 -> b1) -> (a -> Either b a1) -> (a -> Either b b1)
mapExpect f g x = case g x of
                           (Left l) => Left l
                           (Right r) => Right (f r)

instance Functor (Validation b a) where 
    map f (Expect g) = Expect $ f `mapExpect` g


appExpect : (f : a -> Either b (a1 -> b1)) -> (g : a -> Either b a1) -> a -> Either b b1
appExpect f g x = case f x of
                       (Left l) => Left l
                       (Right fr) => (case g x of
                                          (Left l) => Left l
                                          (Right r) => Right $ fr r)

instance Applicative (Validation b a) where
    pure x = Expect (\_ => Right x)
    (<*>) (Expect f) (Expect g) = Expect $ appExpect f g



expectBind : (g : a -> Either b a1) -> (f : (result : a1) -> Validation b a b1) -> a -> Either b b1
expectBind g f x = case g x of
                        (Left l) => Left l
                        (Right r) => (case f r of
                                           (Expect f) => f x)

instance Monad (Validation b a) where
    (>>=) (Expect g) f = Expect $ expectBind g f
    
    



hasKey : a -> c -> Validation c (SortedMap a b) b
hasKey x y = Expect (hasKey' x y) where
  hasKey' : a -> c -> SortedMap a b -> Either c b
  hasKey' key error map = case lookup key map of
                               Nothing => Left error
                               (Just x) => Right x






runVal : a -> Validation b a c -> Either b c
runVal x (Expect f) = f x



data HList : List Type -> Type where
  Nil : HList []
  (::) : a -> HList as -> HList (a :: as)
  



x : HList [String, Int]
x = ["jimmy", 2]
