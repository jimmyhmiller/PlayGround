import Effects
import Effect.StdIO
import Effect.State
import Data.Vect
import Effect.System




greet : String -> String
greet x = "Hello " ++ x




hello : Eff () [STDIO]
hello = do putStr "Name? "
           x <- getStr
           putStr $ greet x
           
           
           
           
  
emain : SimpleEff.Eff () [SYSTEM, STDIO]
emain = do [prog, arg] <- getArgs | [] => putStrLn "Can't happen!"
                                  | [prog] => putStrLn "No arguments!"
                                  | _ => putStrLn "Too many arguments!"
           putStrLn $ "Argument is " ++ arg


main : IO ()
main = run hello
