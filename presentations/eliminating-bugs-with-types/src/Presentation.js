
import React from "react";
import { 
  Presentation,
  Code,
  Headline,
  TwoColumn,
  Point,
  Points,
  preloader,
  BlankSlide,
  Image,
  Text,
  formatCode,
} from "./library";

import CodeSlide from 'spectacle-code-slide';

const images = {
  me: require("./images/me.jpg"),
};

preloader(images);

export default () =>
  <Presentation>
    <Headline
      textAlign="left"
      subtextSize={4}
      subtextCaps={true}
      text="Eliminating Bugs with Types" />

    <TwoColumn
      title="About Me"
      left={<Image src={images.me} />}
      right={
        <div style={{paddingTop: 80}}>
          <Text textColor="blue" textSize={60} textAlign="left">Jimmy Miller</Text>
          <Points noSlide styleContainer={{paddingTop: 10}}>
            <Point textSize={40} text="Self Taught" /> 
            <Point textSize={40} text="Senior Developer - healthfinch" /> 
            <Point textSize={40} text="FP Nerd" />
          </Points>
        </div>
      }
    />

    <Headline
      textAlign="left"
      subtextSize={4}
      subtextCaps={true}
      text="Eliminating Bugs with Types" />

    <Headline
      color="blue"
      text="Types Have gotten a bad name" />

    <Points title="Problem With Types">
      <Point text="Verbosity" />
      <Point text="Inheritence" />
      <Point text="Subtyping" />
    </Points>

    <Points title="Uses of Types">
      <Point text="Catching Mistakes" />
      <Point text="Making Programs Run Faster" />
      <Point text="Enforcing Invariants" />
    </Points>

    <Points title="Two Views of Types">
      <Point text="Types for modeling our problem" />
      <Point text="Types for fewer implementations" />
    </Points>

    <Code
      title="Example"
      lang="elm"
      source={`
        type alias Model =
            { questions : List String
            , answers : List (Maybe String)
            }
      `} />

    <Code
      title="Example"
      lang="elm"
      source={`
        init : Model
        init =
            Model [ "What is this?" ] [ Just "Something" ]
      `} />

    <Code
      title="Example"
      lang="elm"
      source={`
        init : Model
        init =
            Model [ "What is this?" ] []
      `} />

    <Code
      title="Example"
      lang="elm"
      source={`
        type alias Model =
            List
                { question : String
                , answer : Maybe String
                }
      `} />

    <Code
      title="Example"
      lang="elm"
      source={`
        init : Model
        init =
            [ { question = "What is this?", answer = Just "Something" } ]
      `} />

    <Code
      title="Example"
      lang="elm"
      source={`
        init : Model
        init =
            []
      `} />

    <Code
      title="Example"
      lang="elm"
      source={`
        type NonEmptyList a
            = NonEmpty a (List a)

        type alias Model =
            NonEmptyList
                { question : String
                , answer : Maybe String
                }
      `} />

    <Code
      title="Example"
      lang="elm"
      source={`
        init : Model
        init =
          NonEmpty { question = "What is this?", answer = Just "Something" } []
      `} />

    <Headline
      textAlign="left"
      text="Illegal States are unrepresentable" />

    <Code
      title="Example"
      lang="elm"
      source={`
        type alias SurveyQuestion =
            { question : String, answer : Maybe String }

        type alias Model =
            { questions : NonEmptyList SurveyQuestion, currentQuestion : Int }

      `} />

    <Code
      title="Example"
      lang="elm"
      source={`
        init : Model
        init =
            { questions =
                NonEmpty { question = "What is this?", 
                           answer = Just "Something" } []
            , currentQuestion = 0
            }
      `} />

    <Code
      title="Example"
      lang="elm"
      source={`
        init : Model
        init =
            { questions =
                NonEmpty { question = "What is this?", 
                           answer = Just "Something" } []
            , currentQuestion = 1
            }
      `} />


    <Code
      title="Example"
      lang="elm"
      source={`
        type ZipperList a
            = Zipper
                { left : List a
                , focus : a
                , right : List a
                }

        type alias Model =
            { questions : ZipperList SurveyQuestion }
      `} />


    <Code
      title="Example"
      lang="elm"
      source={`
        init : Model
        init =
            { questions =
                Zipper
                    { left = [ { question = "What is this?", 
                                 answer = Just "Something" } ]
                    , focus = { question = "Second Question", 
                                answer = Just "Second Answer" }
                    , right = []
                    }
            }
      `} />

      <Headline
        textAlign="left"
        text="Illegal States are unrepresentable" />


    <BlankSlide />

  </Presentation>
