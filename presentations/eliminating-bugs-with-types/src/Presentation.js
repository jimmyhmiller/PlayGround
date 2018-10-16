
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
      size={4}
      text="Using Types to Make (many) Bugs Impossible" />

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
      size={4}
      text="Using Types to Make (many) Bugs Impossible" />

    <Headline
      color="green"
      text="Not a talk on advanced type hackery" />

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

    <Points title="Bugs In Prod">
      <Point text="Almost never simple type errors" />
      <Point text="Always around edge cases" />
      <Point text="Usually introduced late" />
      <Point text="Often introduced by others" />
    </Points>

    <Headline
      color="cyan"
      text="Bugs Happen when Intent isn't clear" />

    <Headline
      size={4}
      subtextSize={5}
      caps={false}
      text="Programs must be written for people"
      subtext="and only incidentally for machines" />

    <Headline
      size={4}
      color="blue"
      text="Your types should guide people to the right answer" />

    <Headline
      size={4}
      subtextSize={6}
      color="green"
      text="Example - Survey Application" />

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
          NonEmpty { question = "What is this?"
                   , answer = Just "Something" } []
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
            { questions : NonEmptyList SurveyQuestion
            , currentQuestion : Int }

      `} />

    <Code
      title="First Question Selected"
      lang="elm"
      source={`
        init : Model
        init =
            { questions =
                NonEmpty { question = "What is this?" 
                         , answer = Just "Something" } []
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
                NonEmpty { question = "What is this?"
                         , answer = Just "Something" } []
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
                    { left = [ { question = "What is this?" 
                               , answer = Just "Something" } ]
                    , focus = { question = "Second Question"
                              , answer = Just "Second Answer" }
                    , right = []
                    }
            }
      `} />

    <Headline
      textAlign="left"
      text="Illegal States are unrepresentable" />

    <Headline
      size={4}
      subtextSize={6}
      color="green"
      text="Example - User Data Model" />

    <Code
      title="User Data Model"
      lang="elm"
      source={`
        type alias UserLogin =
            { username: String
            , password: String
            }
            
        type alias ThirdPartyLogin =
            { apiToken: String
            , email: String
            }
            
        type alias User =
            { name: String
            , userLogin: Maybe UserLogin
            , thirdPartyLogin: Maybe ThirdPartyLogin
            }
      `} />

    <Code
      title="Illegal State"
      lang="elm"
      source={`
        user =
            User
                { name = "Jimmy"
                , userLogin = Nothing
                , thirdPartyLogin = Nothing
                }

      `} />

    <Code
      title="User Data Model Alternative 1"
      lang="elm"
      source={`
        type alias UserLogin =
            { name: String
            , username: String
            , password: String
            }
            
        type alias ThirdPartyLogin =
            { name: String
            , apiToken: String
            , email: String
            }
            
        type User
            = LoginWithUser UserLogin
            | LoginThirdParty ThirdPartyLogin
            | LoginBoth UserLogin ThirdPartyLogin
      `} />

    <Code
      title="User Data Model Alternative 2"
      lang="elm"
      source={`
        type alias UserLogin =
            { username : String
            , password : String
            }

        type alias ThirdPartyLogin =
            { apiToken : String
            , email : String
            }

        type Login
            = User UserLogin
            | ThirdParty ThirdPartyLogin
            | Both UserLogin ThirdPartyLogin

        type alias User = { name : String, login : Login }
      `} />

    <Code
      title="These"
      lang="elm"
      source={`
        type These a b
            = This a
            | That b
            | These a b

        x : These String Int
        x = This "String"

        y : These String Int
        y = That 2

        z : These String Int
        z = These "String" 2
      `} />

    <Code
      title="User Data Model Alternative 3"
      lang="elm"
      source={`
        type alias UserLogin =
            { username : String
            , password : String
            }

        type alias ThirdPartyLogin =
            { apiToken : String
            , email : String
            }

        type alias User =
            { name : String
            , login : These UserLogin ThirdPartyLogin
            }
      `} />

    <Code
      title="Reusable Functions"
      lang="elm"
      source={`
        getUniqueId : These UserLogin ThirdPartyLogin -> String
        getUniqueId = these .username .email (\\u _ -> u.username)
      `} />


    <Headline
      color="blue"
      size={4}
      text="The Types of our functions communicate intent" />

    <Code
      title="Uninteresting types"
      lang="java"
      source={`
        public String greet(String name);
      `} />

    <Code
      title="Uninteresting types"
      lang="java"
      source={`
        public String asdf(String zxcv);
      `} />

    <Code
      title="Kinda Communicating with types"
      lang="java"
      source={`
        public interface Thing<E> {
          boolean stuff();
          E do();
          void go();
        }
      `} />

    <Code
      title="Kinda Communicating with types"
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
      size={5}
      caps={false}
      subtextSize={5}
      text="The less our program tells us about our types"
      subtext="The more it tells us about our code" />

    <Code
      title="Communicating with types"
      lang="haskell"
      source={`
        f : String -> String
      `} />

    <Code
      title="Communicating with types"
      lang="haskell"
      source={`
        f : a -> a
      `} />

    <Code
      title="Communicating with types"
      lang="haskell"
      source={`
        f : List Char -> List Char -> List Char
      `} />

    <Code
      title="Communicating with types"
      lang="haskell"
      source={`
        f : List a -> List a -> List a
      `} />

    <Code
      title="Communicating with types"
      lang="haskell"
      source={`
        f : Semigroup a => a -> a -> a
      `} />

    <Headline
      text="Using generic types makes more programs impossible" />

    <Headline
      color="green"
      text="Types fundamentally limit" />

    <Headline
      color="cyan"
      text="The set of all sets that don't contain themselves" />

    <Headline
      text="Russell Proposed types in 1902" />

    <Headline
      size={5}
      caps={false}
      subtextSize={5}
      text="In order to use types to make bugs impossible"
      subtext="Use types to make programs impossible" />

    <BlankSlide />

  </Presentation>
