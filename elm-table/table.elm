
import Html (..)
import Html.Attributes (..)
import Html.Events (..)
import Signal
import Window
import Array (Array, get, set, fromList, toList)
import Array
import List
import List ((::))
import Graphics.Element (Element, container, midTop)

type alias Data  = {name : String, age : String}

type alias State = {columnMappings : Array (ColumnMapping Data),
                    data : List Data}

type alias ColumnMapping a = {heading : String, field : (a -> String)}

startingState : State
startingState = {columnMappings = fromList [{heading = "Name", field = .name}, 
                                            {heading =  "Age", field = .age}],
                data = [{name="jimmy", age="22"}, 
                        {name="janice", age="26"}]}

swap : Int -> Int -> Array a -> Array a
swap a b coll = case (get a coll) of
    Just x -> case (get b coll) of
        Just y ->  set b x (set a y coll)
        Nothing -> coll
    Nothing -> coll

type Update = NoOp
            | ReorderColumn Int Int

step : Update -> State -> State
step act state = 
    case act of 
        NoOp -> state
        ReorderColumn a b -> {state | columnMappings <- swap a b state.columnMappings}

state : Signal.Signal (State)
state = Signal.foldp step startingState (Signal.subscribe updates)

updates : Signal.Channel Update
updates = (Signal.channel NoOp)

scene : State -> (Int,Int) -> Element
scene state (w,h) = container w h midTop (toElement 550 h (view state))

view : State -> Html
view {columnMappings, data} = 
    div [] [a 
                [onClick (Signal.send updates (ReorderColumn 0 1))] 
                [text "reorder"], 
            elmTable columnMappings data]

elmTable : Array (ColumnMapping a) -> List a -> Html
elmTable clms data = table [] [tableHead clms,
                               tableBody clms data]

tableHead : Array (ColumnMapping a) -> Html
tableHead clms = thead [] 
                    [tr [] 
                        (toList 
                            (Array.map 
                                (\x -> th [] [text x.heading]) 
                                clms))]

tableBody : Array (ColumnMapping a) -> List a -> Html
tableBody clms data = tbody [] (List.map (tableRow clms) data)

tableRow : Array (ColumnMapping a) -> a -> Html
tableRow clms data = tr 
                        [] 
                        (toList 
                            (Array.map 
                                (\f -> td [] [text (f.field data)]) 
                                clms))

list : List String -> Html
list coll = ul [] (List.map (\i -> li [] [text i]) coll)

main : Signal.Signal Element
main = Signal.map2 scene state Window.dimensions