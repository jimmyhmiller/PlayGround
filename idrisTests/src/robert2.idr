data Format = CInt Format | CString Format | COther Char Format | CEnd


class From a b where
  from : a -> b


charsToFormat : List Char -> Format
charsToFormat [] = CEnd
charsToFormat ('%' :: 's' :: xs) = CString $ charsToFormat xs
charsToFormat ('%' :: 'd' :: xs) = CInt $ charsToFormat xs
charsToFormat (x :: xs) = COther x $ charsToFormat xs


instance Cast String Format where
    cast x = charsToFormat (unpack x)

instance Cast (List Char) Format where
  cast x = charsToFormat x
  
instance Cast x x where
    cast x = x
    
instance Cast x (List x) where
  cast x = [x]

instance Cast Int (List Char) where
    cast orig = unpack $ show orig



class (Monoid s, Cast Int s, Cast String s, Cast Char s, Cast s Format) => Printfable s where 
instance(Monoid s, Cast Int s, Cast String s, Cast Char s, Cast s Format) => Printfable s where 



formatToType : Semigroup s => Format -> Type
formatToType (CInt rest) {s} = Int -> formatToType {s} rest
formatToType (CString rest) {s} = String -> formatToType {s} rest
formatToType (COther c rest) {s} = formatToType {s} rest
formatToType CEnd {s} = s

formatToFormatType : Printfable s => (f : Format) -> (acc : s) -> formatToType {s} f
formatToFormatType (CInt rest) acc = (\i => formatToFormatType rest (acc <+> cast i))
formatToFormatType (CString rest) acc = (\s => formatToFormatType rest (acc <+> cast s))
formatToFormatType (COther c rest) acc = formatToFormatType rest (acc <+> cast c)
formatToFormatType CEnd acc = acc

printf : Printfable s => (pattern : s) -> formatToType {s} $ cast pattern
printf pattern {s} = formatToFormatType {s} (cast pattern) neutral


greeting : String -> List Char
greeting = printf (unpack "hello %s")




--printf "hello %d" : Int -> String
--printf "hello %s %d" : String -> Int -> String
--printf "hello" : String 
