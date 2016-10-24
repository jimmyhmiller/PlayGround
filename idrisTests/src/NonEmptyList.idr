

data NEL a = Cons a (Maybe (NEL a))

x : NEL Int
x = Cons 1 Nothing
