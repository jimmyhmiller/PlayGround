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

formatToFormatType : (f : Format) -> (acc : String) -> formatToType f
formatToFormatType (CInt y) acc = (\i => formatToFormatType y (acc ++ show i))
formatToFormatType (CString y) acc = (\s' => formatToFormatType y (acc ++ s'))
formatToFormatType (COther c y) acc = formatToFormatType y (acc ++ singleton c) 
formatToFormatType CEnd acc = acc

printf : (s : String) -> formatToType (charsToFormat (unpack s))
printf s = formatToFormatType (charsToFormat (unpack s)) ""

greeting : String -> Int -> String
greeting = printf "Hello %s %d"
