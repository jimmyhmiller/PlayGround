import Browser
import Html exposing (..)
import Html.Attributes exposing (..)
import Html.Events exposing (onInput)



-- MAIN


main =
  Browser.sandbox { init = init, update = update, view = view }



-- MODEL


type alias Model =
  { questions : List String
  , answers : List (Maybe String)
  }


init : Model
init =
  Model ["What is this?"] []



-- UPDATE


type Msg
  = AddQuestion String


update : Msg -> Model -> Model
update msg model = model


-- VIEW

viewQuestion : String -> Html msg
viewQuestion question = div [] [h1 [] [(text "Question")], (text question)]


view : Model -> Html Msg
view model =
  div [] (List.map viewQuestion model.questions)


    --[ viewInput "text" "Name" model.name Name
    --, viewInput "password" "Password" model.password Password
    --, viewInput "password" "Re-enter Password" model.passwordAgain PasswordAgain
    --, viewValidation model
    --]


--viewInput : String -> String -> String -> (String -> msg) -> Html msg
--viewInput t p v toMsg =
--  input [ type_ t, placeholder p, value v, onInput toMsg ] []


--viewValidation : Model -> Html msg
--viewValidation model =
--  if model.password == model.passwordAgain then
--    div [ style "color" "green" ] [ text "OK" ]
--  else
--    div [ style "color" "red" ] [ text "Passwords do not match!" ]
