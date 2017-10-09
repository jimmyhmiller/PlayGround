
module Main where
import System.IO
import qualified Data.ByteString.Lazy as B
import Data.Binary.Get
import Data.Word
import Data.Int
-- import Data.Vector.Unboxed

(|>) :: a -> (a -> b) -> b
(|>) a f = f a

skipToFirstTag :: B.ByteString -> B.ByteString
skipToFirstTag b = 
    B.dropWhile (\b -> b /= 0) b
    |> B.drop 1
    |> B.drop 4
    |> B.drop 8

skipToNextTag :: Get [Int64]
skipToNextTag = do
    empty <- isEmpty
    if empty
        then return []
        else do
            skip 5
            length <- getWord32be
            skip (fromIntegral length)
            rest <- skipToNextTag
            location <- bytesRead
            return (location : rest)


main = do
    file <- B.readFile "/Users/jimmy.miller/Desktop/largedump.hprof"
    let firstTag = skipToFirstTag file
    let x = runGet skipToNextTag firstTag

    putStrLn $ show $ x
    -- withBinaryFile "/Users/jimmy.miller/Desktop/largedump.hprof" ReadMode $ \handle -> do
    --     hSeek handle RelativeSeek 31
    --     pos <- hTell handle;
    --     by <- B.hGet handle 1
    --     putStrLn $ show $ B.head by
    --     hSeek handle RelativeSeek 31