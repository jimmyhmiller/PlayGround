data Natural = Zero | Successor Natural


zero : Natural
zero = Zero

one : Natural
one = Successor zero

two : Natural
two = Successor one


inc : Natural -> Natural
inc x = Successor x


plus : Natural -> Natural -> Natural
plus Zero Zero = Zero
plus Zero x = x
plus x Zero = x
plus x (Successor y) = inc x `plus` y
