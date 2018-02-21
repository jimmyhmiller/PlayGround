data Door = Opened | Closed

data DoorCmd : Type -> Door -> Door-> Type where
     Open : DoorCmd () Closed Opened
     Close : DoorCmd () Opened Closed
     Knock : DoorCmd () Closed Closed

     Pure : a -> DoorCmd a state state
     (>>=) : DoorCmd a state1 state2 -> 
       (a -> DoorCmd b state2 state3) -> 
       DoorCmd b state1 state3


doorProg : DoorCmd () Closed Closed
doorProg = do Knock
              Open
              Close
