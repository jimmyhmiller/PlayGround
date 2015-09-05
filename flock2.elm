import List exposing (foldr, length, (::), sum, sortWith, filter)
import Graphics.Collage exposing (rotate, filled, polygon, collage, move, toForm)
import Color exposing (black)
import Signal exposing (map, map2, foldp)
import Time exposing (fps)
import List as List
import Debug as Debug

type Vector = Vector Float Float


add (Vector x y) (Vector x' y') = Vector (x + x') (y + y')
dot (Vector x y) (Vector x' y') = Vector (x * x') (y * y')

magnitude (Vector x y) = sqrt(x^2 + y^2)
rotation (Vector x y) = Debug.watch "rotation" (atan2 y x)


rotate' (Vector x y) (Vector x' y') = Vector (x * y' - y * x') (x * x' + y * y')

degreetoVector degree = Vector (xcomp (radians degree)) (ycomp (radians degree)) |> Debug.watch "convert"

xcomp r = sin r
ycomp r = cos r



drawSheep s = (polygon [(10,0),(-10,10),(-10,-10)])
       |> filled black 
       |> move (s.x, s.y)
       |> rotate (rotation s.v)


delta = map (\t -> t/20) (fps 25)

x (Vector x _) = x
y (Vector _ y) = y

movesheep d s = {s | 
                     x <- s.x + x s.v*d,
                     y <- s.y + y s.v*d}
                |> Debug.watch "sheep" 


turn angle s = add angle s

sheep = {x=20, y=20, v=Vector 0 -1 }

toList s = [s]

main = foldp movesheep sheep delta
      |> map drawSheep
      |> map toList
      |> map (collage 300 300)


--main : Signal Element
--main = foldp movesheeps createSheep delta
--    |> map (List.map drawSheep)
--    |> map (collage 500 500)

 