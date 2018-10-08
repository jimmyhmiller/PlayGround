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


type ZipperList a
    = Zipper
        { left : List a
        , focus : a
        , right : List a
        }


type alias SurveyQuestion =
    { question : String, answer : Maybe String }


type alias Model =
    { questions : ZipperList SurveyQuestion }


init : Model
init =
    { questions =
        Zipper
            { left = [ { question = "What is this?", answer = Just "Something" } ]
            , focus = { question = "Second Question", answer = Just "Second Answer" }
            , right = []
            }
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


viewQuestionAnswer : { question : String, answer : Maybe String } -> Html msg
viewQuestionAnswer { question, answer } =
    div []
        [ viewQuestion question, viewAnswer answer ]


viewCurrent : ZipperList SurveyQuestion -> Html msg
viewCurrent (Zipper { left, focus, right }) =
    viewQuestionAnswer focus


view : Model -> Html Msg
view { questions } =
    div [] [ viewCurrent questions ]
