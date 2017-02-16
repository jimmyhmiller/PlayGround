andThen : IO () -> IO () -> IO ()
andThen x y = do
  y
  x
  
justBefore : IO () -> IO () -> IO ()
justBefore x y = do
  x
  y

x : List (IO ())
x = [putStr "hello", putStr "world"]

y : List (IO ())
y = map (andThen (putStr " then ")) x

z : List (IO ())
z = map (justBefore (putStr " before ")) y


main : IO ()
main = head z
