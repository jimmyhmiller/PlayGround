import Effect.Default

data Write = Written | Bottom


data CloudType : Type -> Type where
  CInt : Bool -> Int -> Int -> CloudType Int
  CString : Write -> String -> CloudType String




Default (CloudType Int) where
    default = CInt False 0 0
    
Default (CloudType String) where
    default = CString Bottom ""

interface Cloud (c : Type -> Type) where
  init : c a
  get : c a -> a
  set : a -> c a -> c a
  fork : c a -> (c a, c a)
  join : c a -> c a -> c a

Cloud CloudType where
    init = ?test
    get (CInt r b d) = b + d
    set n (CInt x y z) = CInt True n 0
    fork (CInt r b d) = (CInt r d b, CInt False (b + d) 0)
    join (CInt r b d) (CInt r' b' d') = case r of
                                             False => CInt r b (d + d')
                                             True => CInt True b' d'


-- add : (n : Int) -> CInt -> CInt
-- add n (I r b d) = I r b (d + n)

-- set : (n : Int) -> CInt -> CInt
-- set n (I r b d) = I True n 0

-- get : CInt -> Int
-- get (I r b d) = b + d

-- fork : CInt -> (CInt, CInt)
-- fork (I r b d) = (I r d b, I False (b + d) 0)

-- join : CInt -> CInt -> CInt
-- join (I r b d) (I r' b' d') = case r of
--                                    False => I r b (d + d')
--                                    True => I True b' d'

