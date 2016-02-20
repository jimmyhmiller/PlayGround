import Data.List

data Json : List String -> Type where
  Empty : Json []
  Add : (key : String) -> (value: String) -> Json ks -> Json (key :: ks)



getValue : (key : String) -> Json ks -> {auto ok: Elem key ks} -> String
getValue key (Add key value x) {ok = Here} = value
getValue key (Add y value x) {ok = (There z)} = getValue key x


addEm : Int -> Int -> Int -> Int
addEm x y z = x + y + z



findValue : (n : Int) -> (coll : List Int) -> {auto ok: Elem n coll} -> Int 
findValue n (n :: xs) {ok = Here} = 0
findValue n (y :: xs) {ok = (There x)} = 1 + (findValue n xs)


q : Json ["name", "age"]
q = Add "name" "jimmy" $
    Add "age" "24" Empty

    
addOtherName : String -> Json ["name","age"] -> Json ["otherName","name", "age"]
addOtherName x y = Add "otherName" x y


othername : String
othername = getValue "otherName" $ addOtherName "James" q







--data Json : Type where
--  JsonString : String -> Json
--  JsonNumber : Double -> Json
--  JsonObject : List (String, Json) -> Json
  
 
