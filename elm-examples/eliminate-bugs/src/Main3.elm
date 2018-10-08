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


type NonEmptyList a
    = NonEmpty a (List a)


map : (a -> b) -> NonEmptyList a -> NonEmptyList b
map f (NonEmpty a rest) =
    NonEmpty (f a) (List.map f rest)


type alias Model =
    NonEmptyList
        { question : String
        , answer : Maybe String
        }


init : Model
init =
    NonEmpty { question = "What is this?", answer = Just "Something" } []


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


viewQuestionAnswer : { question : String, answer : Maybe String } -> Html msg
viewQuestionAnswer { question, answer } =
    div []
        [ viewQuestion question, viewAnswer answer ]


view : Model -> Html Msg
view (NonEmpty first rest) =
    div [] ([ viewQuestionAnswer first ] ++ List.map viewQuestionAnswer rest)
