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


get : Int -> NonEmptyList a -> Maybe a
get i (NonEmpty first rest) =
    case i of
        0 ->
            Just first

        n ->
            List.head (List.drop (n - 1) rest)


type alias SurveyQuestion =
    { question : String, answer : Maybe String }


type alias Model =
    { questions : NonEmptyList SurveyQuestion, currentQuestion : Int }


init : Model
init =
    { questions =
        NonEmpty { question = "What is this?", answer = Just "Something" }
            [ { question = "Second Question", answer = Just "Second Answer" } ]
    , currentQuestion = 1
    }


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


viewQuestionAnswer : Maybe { question : String, answer : Maybe String } -> Html msg
viewQuestionAnswer survey =
    case survey of
        Just { question, answer } ->
            div []
                [ viewQuestion question, viewAnswer answer ]

        Nothing ->
            div [] []


view : Model -> Html Msg
view { questions, currentQuestion } =
    div [] [ viewQuestionAnswer (get currentQuestion questions) ]
