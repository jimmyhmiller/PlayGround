data Format = CInt Format | CString Format | COther Char Format | CEnd

charsToFormat : List Char -> Format
charsToFormat [] = CEnd
charsToFormat ('%' :: 's' :: xs) = CString $ charsToFormat xs
charsToFormat ('%' :: 'd' :: xs) = CInt $ charsToFormat xs
charsToFormat (x :: xs) = COther x $ charsToFormat xs

formatToType : Format -> Type
formatToType (CInt x) = Int -> formatToType x
formatToType (CString x) = String -> formatToType x
formatToType (COther c x) = formatToType x
formatToType CEnd = String

formatToFormatType : (f : Format) -> String -> formatToType f
formatToFormatType (CInt y) s = (\i => formatToFormatType y (s ++ show i))
formatToFormatType (CString y) s = (\s' => formatToFormatType y (s ++ s'))
formatToFormatType (COther c y) s = formatToFormatType y (s ++ singleton c) 
formatToFormatType CEnd s = s

printf : (s : String) -> formatToType (charsToFormat (unpack s))
printf s = formatToFormatType (charsToFormat (unpack s)) ""

greeting : String -> Int -> String
greeting = printf "Hello %s %d"
