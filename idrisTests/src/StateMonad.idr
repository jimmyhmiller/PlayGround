data STATE s a = State (s -> (a, s))


implementation Functor (STATE s) where
  map func (State f) = State (\s => (case f s of
                                          (a, b) => (func a, b)))


implementation Applicative (STATE s) where
  pure x = State (\s => (x, s))
  (<*>) (State f) (State g) = State (\s => (case f s of
                                                 (f, s1) => (case g s of
                                                                  (a, b) => (f a, b))))


implementation Monad (STATE s) where
  (>>=) (State f) k = State (\st => (case f st of
                                        (v, st') => (case k v of
                                                        (State kv) => kv st')))



get : STATE s s
get = State (\s => (s, s))


put : s -> STATE s ()
put x = State (\_ => ((), x))


runState : STATE s a -> s -> a
runState (State f) s = case f s of
                            (a, b) => a



inc : STATE Int Int
inc = do 
  x <- get
  put $ x + 1
  pure $ x

inc3 : STATE Int Int
inc3 = do
  inc
  inc
  inc



my_pow : Nat -> Nat -> Nat
my_pow x Z = 1
my_pow x (S k) = mult x (my_pow x k)

pow2 : Nat -> Nat
pow2 k = my_pow k 2


fact' : Nat -> Nat
fact' Z = 1
fact' (S k) = (S k) * fact' k
