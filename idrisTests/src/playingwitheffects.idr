import Effects
import Effect.StdIO
import Effect.State
import Data.Vect
import Effect.System





emain : SimpleEff.Eff () [SYSTEM, STDIO]
emain = do [prog, arg] <- getArgs | [] => putStrLn "Can't happen!"
                                  | [prog] => putStrLn "No arguments!"
                                  | _ => putStrLn "Too many arguments!"
           putStrLn $ "Argument is " ++ arg


main : IO ()
main = run emain
