import Effect.StdIO

identity : a -> a
identity x = x


data Optional a = None | Something a




data Vect : Nat -> Type -> Type where
  Nil : Vect 0 a
  (::) : a -> Vect n a -> Vect (S n) a


Functor Optional where
    map func None = None
    map func (Something x) = Something (func x)


append : Vect n a -> Vect m a -> Vect (n + m) a
append [] y = y
append (x :: z) y = x :: append z y

x : Vect 1 Int
x = [1]



hello : Eff () [STDIO]
hello = do putStr "Name? "
           x <- getStr
           putStrLn ("Hello " ++ trim x ++ "!")
