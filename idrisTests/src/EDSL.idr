data ClickEvent = Click String String
data PlayEvent = Play String Integer
data PauseEvent = Pause String Integer Integer


data HList : List Type -> Type where
  Nil : HList []
  (::) : t -> HList ts -> HList (t :: ts)


Events : Type
Events = HList [ClickEvent, PlayEvent, PauseEvent]


interface Named a where
  name : String


Named ClickEvent where
    name = "click"

Named PlayEvent where
    name = "play"
    
Named PauseEvent where
    name = "pause"
    
Named (HList []) where
    name = ""


(Named e, Named (HList tail)) => Named (HList (e :: tail)) where
    name {e} {tail} = (Main.name {a=e}) ++ ", " ++ (Main.name {a=HList tail})

interface HandleEvent e where
  Out : Type
  handleEvents : (event : String) -> (payload : String) -> Either String Out
  

HandleEvent (HList []) where
    Out = HList []
    handleEvents event payload = Left $ "Could not handle " ++ event ++ "with payload " ++ payload 
    

 
 interface FromString a where
   fromString : String -> Maybe a
   
   
FromString ClickEvent where
  fromString x = Just $ Click "test" "test"
 
 
 FromString PlayEvent where
   fromString x = Just $ Play "test" 2
 
FromString PauseEvent where
   fromString "pause" = Just $ Pause "test" 2 1
   fromString x = Nothing
 
 
 eventFromString : String -> Type
 eventFromString "click" = ClickEvent
 eventFromString "pause" = PauseEvent
 eventFromString "play" = PlayEvent
 
(FromString e, Named e, HandleEvent (HList tail)) => HandleEvent (HList (e :: tail)) where
    Out {tail} {e} = Either (Out {e=(HList tail)}) e
    handleEvents event payload {e} {tail} = let headName = Main.name {a=e} in
      (case event == headName of
            True => (case fromString {a=e} payload of
                          Nothing => Left $ "Could not decode " ++ event ++ " from " ++ payload
                          (Just x) => Right (Right x))
            False => map Left $ handleEvents {e=(HList tail)} event payload)
