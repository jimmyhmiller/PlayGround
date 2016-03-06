import Data.SortedMap

data Json : Type where
  Str : String -> Json
  Number : Double -> Json
  Boolean : Bool -> Json
  Null : Json
  Array : List Json -> Json
  Object : (SortedMap String Json) -> Json
  
data IsObjectType : Json -> Type where
  IsObject : IsObjectType (Object m)




data Contains : (s : String) -> (m : SortedMap String Json) -> Type where
  Does : Contains s m


get : (s : String) -> (j : Json) -> {auto ok: IsObjectType j} -> Maybe Json
get s {ok = IsObject} (Object m) = lookup s m



q : SortedMap String Json
q = fromList [("name", Str "jimmy")]





contains : (key : a) -> SortedMap a b -> Bool
contains key x = case lookup key x of
                      Nothing => False
                      (Just x) => True

