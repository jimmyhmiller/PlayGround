data Complex = C Integer Integer

instance Eq Complex where
    (==) (C a b) (C c d) = a == c && b == d



instance Num Complex where
    (+) (C a b) (C c d) = C (a + c) (b + d)
    (*) (C a b) (C c d) = C (a * c - b * d) (a * d + b * c)
    fromInteger x = C x 0

i : Complex
i = C 0 1



squared : (n : Complex) -> (n = (C 0 1)) -> pow n 2 = (fromInteger (-1))
squared (C 0 1) Refl = Refl

commute' : (x,z : Integer) -> x + z = z + x
commute' x z = ?test_1



commuteAdd : (w,z : Complex) -> w + z = z + w
commuteAdd (C x y) (C z w) = ?z_rhs_1


