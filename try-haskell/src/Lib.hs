module Lib
    ( someFunc,
      run_x
    ) where

import Data.Promise


someFunc :: IO ()
someFunc = putStrLn "someFunc"

resolve :: a -> Promise s a -> Lazy s ()
resolve a p = p != a

-- getFoo :: Promise s String -> Lazy s ()
-- getFoo = resolve "foo"
--
-- getBar :: Promise s String -> Lazy s ()
-- getBar = resolve "bar"
--
-- getBaz :: Promise s String -> Lazy s ()
-- getBaz = resolve "baz"
--
-- run_x = runLazyIO_ $ x

getFoo :: Maybe String
getFoo = Just "foo"

getBar :: String -> Maybe String
getBar _ = Just "bar"

getBaz :: String -> Maybe String
getBaz _ = Just "baz"

run_x :: IO (Maybe String)
run_x = return x

-- getFoo :: Either String String
-- getFoo = Right "foo"
--
-- getBar :: Either String String
-- getBar = Right "bar"
--
-- getBaz :: Either String String
-- getBaz = Right "baz"
--
-- run_x :: IO (Either String String)
-- run_x = return x


-- getFoo :: IO String
-- getFoo = getLine
--
-- getBar :: IO String
-- getBar = getLine
--
-- getBaz :: IO String
-- getBaz = getLine
--
-- run_x :: IO String
-- run_x = x


x = do
  foo <- getFoo
  bar <- getBar foo
  baz <- getBaz bar
  return baz
