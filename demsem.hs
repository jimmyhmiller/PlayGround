import Prelude hiding (Bool)


type Id = String

data Program = C Command deriving (Eq, Show)
data Command =  (:>) Command Command | If Bool Command | Ife Bool Command Command | Assign Id Expression | Diverge deriving (Eq, Show)
data Expression = Plus Expression Expression| Identifier Id | Numeral Int deriving (Eq, Show)
data Bool = Equal Expression Expression | Not Bool deriving (Eq, Show)


program (C c) = \n -> let s = (update "A" n newstore) in 
    let s' = command c s in access "Z" s'


command ((:>) c1 c2) = \s -> command c2 (command c1 s)
command (If b c) = \s -> if bool b s then command c s else s
command (Ife b c1 c2) = \s -> if bool b s then command c1 s else command c2 s
command (Assign i e) = \s -> update i (expression e s) s
command Diverge = \s -> undefined

expression (Plus e1 e2) = \s -> expression e1 s + expression e2 s
expression (Identifier i) = \s -> access i s
expression (Numeral n) = \s -> n

bool (Equal e1 e2) = \s -> expression e1 s == expression e2 s
bool (Not b) = \s -> not (bool b s)

newstore i = 0
access i s = s i
update i1 n s = \i2 -> if (i1 == i2) then n else s i2

main = putStrLn $ show $ program (C (
                                    (Assign "Z" (Numeral 0) :> 
                                    (If (Equal (Identifier "A") (Numeral 0)) 
                                        Diverge) :> 
                                    (Assign "Z" (Numeral 3)))
                                )) 2
