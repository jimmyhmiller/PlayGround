import Data.Vect




Question : Type
Question = String

Answer : Type
Answer = Maybe String


data Plan = Basic | Premium | Super

questionForPlan : Plan -> Nat
questionForPlan Basic = 1
questionForPlan Premium = 3
questionForPlan Super = 5

dec : Nat -> Nat
dec Z = Z
dec (S k) = k

data QuestionsForPlan : (plan : Plan) -> Type where
  Questions : {n : Nat} -> 
              {auto prf : LTE n (questionForPlan plan)} ->
              Vect n String ->
              QuestionsForPlan plan


qs : QuestionsForPlan Basic 
qs = Questions ["Is this a question"]
 
record User (plan : Plan) where
   constructor MkUser
   name : String
   questions : QuestionsForPlan plan
   
   
j : User Basic
j = MkUser "Jimmy" $
           Questions ["Is this a Question?"]
           
 
newQuestion : String -> (user : User plan) -> (User plan)
newQuestion s {plan = Basic} (MkUser name {plan = Basic} (Questions {n = Z} xs)) = MkUser name (Questions (s :: xs))
newQuestion s {plan = Basic} (MkUser name {plan = Basic} (Questions {n = (S k)} (x :: xs))) = MkUser name (Questions (s :: xs))
newQuestion s {plan = Premium} (MkUser name {plan = Premium} (Questions {n=?test} xs)) = ?newQuestion_rhs_1
newQuestion s {plan = Super} (MkUser name {plan = Super} questions) = ?newQuestion_rhs_4

