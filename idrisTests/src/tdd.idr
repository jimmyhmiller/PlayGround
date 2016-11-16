data Vect : Nat -> Type -> Type where
  Nil : Vect Z a
  (::) : a -> Vect n a -> Vect (S n) a




append : (xs : Vect n a) -> (ys : Vect m a) -> Vect (n + m) a
append [] ys = ys
append (x :: xs) ys = x :: append xs ys

remove : (xs : Vect (S n) a) -> Vect n a
remove (x :: y) = y


defaultReducer : s -> a -> s
defaultReducer x y = x


toAction : (s -> s) -> s -> a -> s
toAction f x y = f x

inc : Int -> Int
inc x = x + 1

dec : Int -> Int
dec x = x - 1

isEven : Int -> Bool
isEven x = mod x 2 == 0


isOdd : Int -> Bool
isOdd x = not $ isEven x


data Action = Increment | Decrement 

reducer' : Int -> Action -> Int
reducer' x Increment = x + 1
reducer' x Decrement = x - 1







initial : s -> (s -> a -> s) -> Maybe s -> a -> s
initial init f Nothing action = f init action
initial _ f (Just state) action = f state action


reducer : (pred : (a -> Bool)) -> (f : (s -> a -> s)) -> (next : (s -> a -> s)) -> (state : s) -> (action : a) -> s
reducer pred f next state action with (pred action)
  reducer pred f next state action | False = next state action
  reducer pred f next state action | True = f state action

evenReducer : (next : Int -> Int -> Int) -> Int -> Int -> Int
evenReducer = reducer isEven (toAction inc)


oddReducer : (next : Int -> Int -> Int) -> Int -> Int -> Int
oddReducer = reducer isOdd (toAction dec)


reduce : Maybe Int -> Int -> Int
reduce = (initial 0 $ evenReducer $ oddReducer const)

