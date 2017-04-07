
data Shape = Oval | Squiggle | Diamond

Eq Shape where
  (==) Oval Oval = True
  (==) Squiggle Squiggle = True
  (==) Diamond Diamond = True
  (==) _ _ = False
  (/=) x y = not (x == y)


data Color = Red | Purple | Green

Eq Color where
  (==) Red Red = True
  (==) Purple Purple = True
  (==) Green Green = True
  (==) _ _ = False
  (/=) x y = not (x == y)


data Shading = Solid | Stripes | Outline

Eq Shading where
  (==) Solid Solid = True
  (==) Stripes Stripes = True
  (==) Outline Outline = True
  (==) _ _ = False
  (/=) x y = not (x == y)



data C : Nat -> Shape -> Color -> Shading -> Type where
  Card : (n : Nat) -> (s : Shape) -> (c : Color) -> (sh : Shading) -> C n s c sh


data ValidSetAttribute : (attr : Type) -> Type where
  AllDifferent : (a : attr) -> (b : attr) -> (c : attr) -> {auto prf1: (a = b -> Void)} -> {auto prf2: (b = c -> Void)} -> ValidSetAttribute attr
  AllSame : (a : attr) -> (b : attr) -> (c : attr) -> {auto prf1: (a = b)} -> {auto prf2: (b = c)} -> ValidSetAttribute attr





data Match : Type where
  SameNumber : C n s1 c1 sh1 -> C n s2 c2 sh2 -> C n s3 c3 sh3 -> Match
  SameShape : C n1 s c1 sh1 -> C n2 s c2 sh2 -> C n3 s c3 sh3 -> Match
  SameColor : C n1 s1 c sh1 -> C n2 s2 c sh2 -> C n3 s3 c sh3 -> Match
  SameShading : C n1 s1 c1 sh -> C n2 s2 c2 sh -> C n3 s3 c3 sh -> Match
  


s : Match
s = SameNumber (Card 0 Oval Red Solid)
               (Card 0 Oval Red Solid)
               (Card 0 Oval Red Solid)
