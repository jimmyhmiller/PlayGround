data Format = CInt Format | CString Format | COther Char Format | CEnd

charsToFormat : List Char -> Format
charsToFormat [] = CEnd
charsToFormat ('%' :: 's' :: xs) = CString $ charsToFormat xs
charsToFormat ('%' :: 'd' :: xs) = CInt $ charsToFormat xs
charsToFormat (x :: xs) = COther x $ charsToFormat xs

formatToType : Format -> Type
formatToType (CInt x) = Int -> formatToType x
formatToType (CString x) = String -> formatToType x
formatToType (COther _ x) = formatToType x
formatToType CEnd = String

formatToFormatType : (f : Format) -> String -> formatToType f
formatToFormatType (CInt rest) s = (\i => formatToFormatType rest (s ++ show i))
formatToFormatType (CString rest) s = (\s' => formatToFormatType rest (s ++ s'))
formatToFormatType (COther c rest) s = formatToFormatType rest (s ++ singleton c)
formatToFormatType CEnd s = s

printf : (s : String) -> formatToType (charsToFormat (unpack s)) 
printf s = formatToFormatType (charsToFormat (unpack s)) ""

greeting : String -> String
greeting x = printf "hello %s" x 



--printf "jimmy %s %d" : String -> Int -> String

--printf "%d" : Int -> String
