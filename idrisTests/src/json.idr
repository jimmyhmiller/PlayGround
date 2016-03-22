import Data.SortedMap
import Data.List


data KeyedMap : (k : Type) -> (v : Type) -> List k -> Type where
  Nil : KeyedMap k v []
  (::) : (key : k) -> (value : v) -> KeyedMap k v xs -> KeyedMap k v (key :: xs)

data Json : Type where
  Str : String -> Json
  Number : Double -> Json
  Boolean : Bool -> Json
  Null : Json
  Array : List Json -> Json
  Object : (KeyedMap String Json xs) -> Json
  
data IsObjectType : Json -> Type where
  IsObject : IsObjectType (Object m)



lookup : (key : k) -> KeyedMap k v xs -> {auto ok : Elem key xs} -> v
lookup key x {ok = Here} = key
lookup key x {ok = (There z)} = ?lookup_rhs_2



get : (s : String) -> (j : Json) -> {auto ok: IsObjectType j} -> Maybe Json
--get s {ok = IsObject} (Object m) = lookup s m


keys : SortedMap k v -> List k
keys x = map fst $ toList x


q : SortedMap String Json
q = fromList [("name", Str "jimmy")]





contains : (key : a) -> SortedMap a b -> Bool
contains key x = case lookup key x of
                      Nothing => False
                      (Just x) => True

