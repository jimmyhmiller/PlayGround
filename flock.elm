import List

--main = collage 1000 1000
--      [(arrow a),
--       (move (30, 0) (arrow b)),
--       (move (50,30) (arrow c)),
--       sheep s1,
--       sheep s2,
--       sheep (movesheep 0.0001 s1 [s2])]

s1 = {x=30, y=50, v=Vector 1 1}
s2 = {x=80, y=50, v=Vector 1 1}
a = toVector s1.x s1.y
b = toVector s2.x s2.y
c = plus a b

arrow (Vector m r) = (rotate r (filled black (polygon [(2,0),(6,6),(10,0),(8,0),(8,-8*m),(4,-8*m),(4,0)])))

data Vector = Vector Float Float

xcomp (Vector m r) = m * sin r
ycomp (Vector m r) = m * cos r

plus v v' = toVector (xcomp v + xcomp v') (ycomp v + ycomp v')
sub v v' = toVector (xcomp v - xcomp v') (ycomp v - ycomp v')
mult v s = toVector ((xcomp v)*s) ((ycomp v)*s)
divide v s = toVector ((xcomp v)/s) ((ycomp v)/s)


addMagnitude (Vector m r) m' = Vector (m+m') r
addRotation (Vector m r) r' = Vector m (r+r')

rotation (Vector _ r) = r

avgV s = case s of
    [] -> Vector 0 0
    s -> (foldr plus (Vector 0 0) s) `divide` toFloat (List.length s)

pythag a b = sqrt(a^2 + b^2)
toVector x y = Vector (pythag x y) (atan2 x y)

normalize v = case v of 
    (Vector m r) -> v `divide` m


--main : Signal Element
main = lift (collage 1000 1000) (lift (map sheep) (foldp movesheeps createSheep delta))

first (x::xs) = x

--main = asText (tooClose (first createSheep) createSheep)


maxSpeed (Vector m r) = if | m > 3 -> Vector 3 r
                           | otherwise -> Vector m r

maxdistance = 100
maxsteer = pi/100
separation = 50

createSheep = map (\r -> {x=20, y=20, v=(Vector 1 r)}) (map (\x -> x * 0.5) [0 .. 100])
 

tooClose o s = s 
               |> neighbors o separation 
               |> map (\x -> {d=(distance o x), v=normalize (plus (toVector o.x o.y) (toVector x.x x.y))})
               |> map (\x -> x.v `divide` x.d)
               |> avgV


movefun x v t = x + v*t 

distance s1 s2 = sqrt ((s1.x - s2.x)^2 + (s1.y - s2.y)^2)

takeWhile p s = case s of
  [] -> []
  (x::xs) -> if p x then x :: takeWhile p xs else []

movesheep t o s = {o | x <- bound (movefun o.x (-1 * xcomp o.v) t) 500,
                       y <- bound (movefun o.y (ycomp o.v) t) 500,
                       v <- maxSpeed ((addRotation o.v (steer o s)) `plus` tooClose o s)}
           
movesheeps t s = map (\o -> movesheep t o s) s        
           

compareDistance o x y = compare (distance o x) (distance o y)

avg s = sum s / toFloat (List.length s)

avgHeading s = avg (map (\o -> rotation o.v) s)



steer o s = let heading = avgHeading (neighbors o maxdistance s)
                distance = abs (rotation o.v - heading)
            in if | distance < maxsteer -> distance
                  | rotation o.v < heading -> maxsteer
                  | rotation o.v > heading -> -1 * maxsteer
                  | otherwise -> 0


neighbors o m s = s |> sortWith (compareDistance o) |> filter (\x -> (distance o x) > 0) |> takeWhile (\x -> (distance o x) < m)
                     

bound x y = if | x < -y -> y
               | x > y -> -y
               | otherwise -> x

delta = lift (\t -> t/20) (fps 25)

sheep s = rotate (rotation s.v) (move (s.x, s.y) (filled black (polygon [(-10,-10),(0,20),(10,-10)])))


