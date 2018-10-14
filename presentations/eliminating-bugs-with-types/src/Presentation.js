
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
      <Point text="Expressing Intent" />
    </Points>

    <Headline
      size={4}
      subtextSize={5}
      caps={false}
      text="Programs must be written for people"
      subtext="and only incidentally for machines" />

    <Headline
      text="Types are meant to communicate" />

    <Code
      title="Communicating with types"
      lang="java"
      source={`
        public String greet(String name);
      `} />

    <Code
      title="Communicating with types"
      lang="java"
      source={`
        public interface Iterator<E> {
          boolean hasNext();
          E next();
          void remove();
        }
      `} />

    <Code
      title="Not Communicating with types"
      lang="java"
      textSize={20}
      source={`
        public interface Functor<F extends K1, Mu extends Functor.Mu> extends Kind1<F, Mu> {
            static <F extends K1, Mu extends Functor.Mu> Functor<F, Mu> unbox(final App<Mu, F> proofBox) {
                return (Functor<F, Mu>) proofBox;
            }

            interface Mu extends Kind1.Mu {}

            <T, R> App<F, R> map(final Function<? super T, ? extends R> func, final App<F, T> ts);
        }
      `} />

    <Code
      title="Communicating with types"
      lang="haskell"
      source={`
        interface Functor (f : Type -> Type) where
            map : (a -> b) -> f a -> f b
      `} />

    <Headline
      size={2}
      text="Types should communicate what is possible and what isn't" />

    <Code
      title="Which tells us more"
      lang="haskell"
      source={`
        thing1 : String -> String
        thing2 : a -> a

        otherThing1 : List a -> List a
        otherThing2 : Vect n a -> Vect n a
      `} />

    <Code
      title="Which tells us more"
      lang="java"
      source={`
        // java
        class User {
          String getName() {...}
        }

        // elm
        type alias User = { name: String }
      `} />

    <Headline
      size={4}
      color="blue"
      text="Your types should guide people to the right answer" />

    <Headline
      size={4}
      subtextSize={6}
      color="green"
      text="Example - Survey Application"
      subtext="In debted to Richard Feldman for the example" />

    <Code
      title="Defining a data model"
      lang="elm"
      source={`
        type alias Model =
            { questions : List String
            , answers : List (Maybe String)
            }
      `} />

    <Code
      title="Example Survey"
      lang="elm"
      source={`
        init : Model
        init =
            Model [ "What is this?" ] [ Just "Something" ]
      `} />

    <Code
      title="What if our list lengths don't match?"
      lang="elm"
      source={`
        init : Model
        init =
            Model [ "What is this?" ] []
      `} />

    <Code
      title="A better data model"
      lang="elm"
      source={`
        type alias Model =
            List
                { question : String
                , answer : Maybe String
                }
      `} />

    <Code
      title="Same Survey State"
      lang="elm"
      source={`
        init : Model
        init =
            [ { question = "What is this?", answer = Just "Something" } ]
      `} />

    <Code
      title="Can we have no questions in a survey?"
      lang="elm"
      source={`
        init : Model
        init = []
      `} />

    <Code
      title="Defining our own datatype"
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
      title="Always one question"
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
      title="How do keep track of current question?"
      lang="elm"
      source={`
        type alias SurveyQuestion =
            { question : String, answer : Maybe String }

        type alias Model =
            { questions : NonEmptyList SurveyQuestion, currentQuestion : Int }

      `} />

    <Code
      title="First Question Selected"
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
      title="What if current is greater than length?"
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
      title="Making a better datatype"
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
      title="Multiple questions, always one selected"
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
