
-- https://mat.iitm.ac.in/home/asingh/public_html/papers/goedel.pdf

infixr 10 <->
infixr 10 ->>

data Grammar =  Var String | P Grammar | (<->) Grammar Grammar | (->>) Grammar Grammar | Not Grammar | Contra



Eq Grammar where
  (==) Contra Contra = True
  (==) (Var x) (Var y) = x == y
  (==) (P x) (P y) = x == y
  (==) (x <-> z) (y <-> w) = x == y && z == w
  (==) (x ->> z) (y ->> w) = x == y && z == w
  (==) (Not x) (Not y) = x == y
  (==) _ _ = False
  

 
double_negation : Grammar -> Grammar
double_negation (Not (Not x)) = x


contradiction : Grammar -> Grammar -> Grammar
contradiction x y = x ->> (Not x ->> y)


distribution_implication : Grammar -> Grammar
distribution_implication p@(x ->> (y ->> z)) = p ->> ((x ->> y) ->> (x ->> z))
distribution_implication x = x


contraposition : Grammar -> Grammar
contraposition (x ->> (Not y)) = (y ->> Not x)
contraposition (x <-> (Not y)) = (y ->> Not x)

modus_ponens : Grammar -> Grammar -> Grammar
modus_ponens x (x' ->> y) = (case x == x' of
                                  True => y) 


transitive : Grammar -> Grammar -> Grammar
transitive (x ->> y) (y' ->> z) = case y == y' of
                                       True => x ->> z

all_provable : Grammar -> Grammar
all_provable x = P x

provable_distribute : Grammar -> Grammar
provable_distribute p@(P (x ->> y)) = p ->> (P x ->> P y)

double_provable : Grammar -> Grammar
double_provable p@(P x) = p ->> (P (P x))




A : Grammar
A = Var "a"

notProvable : Grammar
notProvable = A <-> Not (P A)

step1 : Grammar
step1 = contraposition notProvable

step2 : Grammar
step2 = all_provable step1
 
step3 : Grammar
step3 = provable_distribute step2

step4 : Grammar
step4 = modus_ponens step2 step3

step5 : Grammar
step5 = double_provable (all_provable A)

step6 : Grammar
step6 = transitive step5 step4

----------------------------
step7 : Grammar
step7 = contradiction A Contra

step8 : Grammar
step8 = all_provable step7

step9 : Grammar
step9 = provable_distribute step8

step10 : Grammar
step10 = modus_ponens step8 step9

step11 : Grammar
step11 = provable_distribute (modus_ponens (all_provable A) step10)

step12 : Grammar
step12 = transitive step10 step11

step13 : Grammar
step13 = distribution_implication step12

step14 : Grammar
step14 = modus_ponens step12 step13

step15 : Grammar
step15 = modus_ponens step6 step14
