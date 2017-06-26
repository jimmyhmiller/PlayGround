import Html exposing (Html, button, div, text, span)
import Html.Events exposing (onClick)


main : Program Never Model Msg
main =
  Html.beginnerProgram
    { model = model
    , view = view
    , update = update
    }
type alias Model = Int

model : Model
model = 0

type Msg
  = Increment
  | Decrement
  | IncrementN Int

update : Msg -> Model -> Model
update msg model =
  case msg of
    Increment ->
      model + 1

    Decrement ->
      model - 1

    IncrementN n ->
      model + n

view : Model -> Html Msg
view model =
  div []
    [ button [ onClick Decrement ] [ text "-" ]
    , span [] [ text (toString model) ]
    , button [ onClick Increment ] [ text "+" ]
    , button [ onClick (IncrementN 2) ] [ text "+2" ]
    ]
