identity : a -> a
identity x = x

data Vect : Nat -> Type -> Type where
  Nil : Vect Z a
  (::) : a -> Vect n a -> Vect (S n) a
  
  
q : Vect 1 Int
q = [1]


head' : Vect (S n) a -> a
head' (x :: y) = x


append : Vect n a -> Vect m a -> Vect (n + m) a
append [] y = y
append (x :: z) [] = x :: append z []
append (x :: z) (y :: w) = x :: append z (y :: w)


total
factorial : Nat -> Nat
factorial Z = 1
factorial (S n) = (S n) * fact n



factorial5 : factorial 5 = 120
factorial5 = Refl


zeroOrMore : Nat -> Type
zeroOrMore Z = String
zeroOrMore n = Bool

valueForNat : (n : Nat) -> zeroOrMore n
valueForNat Z = "zero"
valueForNat (S k) = False


data Format = CInt Format | CString Format | COther Char Format | CEnd


charsToFormat : List Char -> Format
charsToFormat [] = CEnd
charsToFormat ('%' :: 's' :: xs) = CString (charsToFormat xs)
charsToFormat ('%' :: 'd' :: xs) = CInt (charsToFormat xs)
charsToFormat (x :: xs) = COther x (charsToFormat xs)

formatToType : Format -> Type
formatToType (CInt x) = Int -> formatToType x
formatToType (CString x) = String -> formatToType x
formatToType (COther c x) = formatToType x
formatToType CEnd = String


formatToFormatType : (f : Format) -> (acc : String) -> formatToType f
formatToFormatType (CInt y) acc = (\i => formatToFormatType y (acc ++ show i))
formatToFormatType (CString y) acc = (\s' => formatToFormatType y (acc ++ s'))
formatToFormatType (COther c y) acc = formatToFormatType y (acc ++ singleton c) 
formatToFormatType CEnd acc = acc


printf : (s : String) -> formatToType (charsToFormat (unpack s))
printf s = formatToFormatType (charsToFormat (unpack s)) ""


greeting : String -> String
greeting = printf "hello %s"
