data Get a

infixr 8 :<|>
data (:<|>) : a -> b -> Type where 
  (:<|>:) : a -> b -> a :<|> b

infixr 9 :>
data (:>) : a -> b -> Type


data Capture a


