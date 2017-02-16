
andThen : IO () -> IO () -> IO ()
andThen x y = do
  y
  x
  

before : IO () -> IO () -> IO ()
before x y = do
  x
  y


x : List (IO ())
x = [putStr "hello", putStr "world"]

y : List (IO ())
y = map (andThen $ putStr " after") x

z : List (IO ())
z = map (before $ putStr "before ") y




Show (IO ()) where
    show x = "IO"
    showPrec d x = "IO"




main : IO ()
main = putStr $ show $ x
--main = head $ z
