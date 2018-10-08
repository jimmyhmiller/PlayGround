module Main exposing (Model, Msg(..), init, main, update, view, viewAnswer, viewQuestion, viewQuestionAnswer)

import Browser
import Html exposing (..)
import Html.Attributes exposing (..)
import Html.Events exposing (onInput)


main =
    Browser.sandbox { init = init, update = update, view = view }


type Msg
    = AddQuestion String


update : Msg -> Model -> Model
update msg model =
    model


type alias Model =
    { questions : List String
    , answers : List (Maybe String)
    }


init : Model
init =
    Model [ "What is this?" ] [ Just "Something" ]


viewQuestion : String -> Html msg
viewQuestion question =
    div [] [ h3 [] [ text "Question" ], text question ]


viewAnswer : Maybe String -> Html msg
viewAnswer answer =
    case answer of
        Nothing ->
            textarea [] []

        Just a ->
            div [] [ text "Answer: ", text a ]


viewQuestionAnswer : String -> Maybe String -> Html msg
viewQuestionAnswer question answer =
    div []
        [ viewQuestion question, viewAnswer answer ]


view : Model -> Html Msg
view model =
    div [] (List.map2 viewQuestionAnswer model.questions model.answers)
