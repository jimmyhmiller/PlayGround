---- MODULE test ----
EXTENDS Integers, TLC

(* --algorithm foo
variables x = 0;
process cycle \in 1..3
begin
  A:
    x := x + 1;
  B:
    x := 0;
  C:
    assert x < 3;
end process
end algorithm; *)
\* BEGIN TRANSLATION
VARIABLES x, pc

vars == << x, pc >>

ProcSet == (1..3)

Init == (* Global variables *)
        /\ x = 0
        /\ pc = [self \in ProcSet |-> "A"]

A(self) == /\ pc[self] = "A"
           /\ x' = x + 1
           /\ pc' = [pc EXCEPT ![self] = "B"]

B(self) == /\ pc[self] = "B"
           /\ x' = 0
           /\ pc' = [pc EXCEPT ![self] = "C"]

C(self) == /\ pc[self] = "C"
           /\ Assert(x /= 2, "Failure of assertion at line 13, column 5.")
           /\ pc' = [pc EXCEPT ![self] = "Done"]
           /\ x' = x

cycle(self) == A(self) \/ B(self) \/ C(self)

Next == (\E self \in 1..3: cycle(self))
           \/ (* Disjunct to prevent deadlock on termination *)
              ((\A self \in ProcSet: pc[self] = "Done") /\ UNCHANGED vars)

Spec == Init /\ [][Next]_vars

Termination == <>(\A self \in ProcSet: pc[self] = "Done")

\* END TRANSLATION


====
