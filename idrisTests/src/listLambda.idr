Y : ((a -> b) -> a -> b) -> a -> b
Y f = ((\x => x x) (\x => f (\arg => (x x) arg))) 
