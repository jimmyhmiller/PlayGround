data PerformAction : Type where
  Action : PerformAction

data FetchData : Type where
  Fetch : FetchData

data Event : Type -> Type where
  Click : Event a
  KeyPress : Int -> Event a

foo : Event a -> Type
foo Click = PerformAction
foo (KeyPress x) = FetchData


withEvent : (e : Event a) -> foo e
withEvent Click = Action
withEvent (KeyPress x) = Fetch
