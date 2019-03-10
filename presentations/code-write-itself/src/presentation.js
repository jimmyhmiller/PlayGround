// Import React
import React from 'react';


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


// Require CSS
require('normalize.css');

const images = {
  me: require("./images/me.jpg"),
};


preloader(images);
require("./langs")

export default () =>
  <Presentation>
    <Headline
      textAlign="left"
      text="Let your code write itself" />

    <TwoColumn
      title="About Me"
      left={<Image src={images.me} />}
      right={
        <div style={{paddingTop: 80}}>
          <Text textColor="blue" textSize={60} textAlign="left">Jimmy Miller</Text>
          <Points noSlide styleContainer={{paddingTop: 10}}>
            <Point textSize={40} text="Self Taught" /> 
            <Point textSize={40} text="Senior Developer - Adzerk" /> 
            <Point textSize={40} text="FP Nerd" />
          </Points>
        </div>
      } 
    />

    <Headline
      textAlign="left"
      text="Let your code write itself" />

    <Headline
      textAlign="left"
      color="blue"
      text="Not a gimmick" />

    <Points title="Approach">
      <Point text="Idris - Static" />
      <Point text="Barliman - Dynamic" />
      <Point text="Both experimental" />
    </Points>

    <Points title="Why?">
      <Point text="Writing Code is Error prone" />
      <Point text="Waste of time" />
      <Point text="This is what programming is about" />
    </Points>

    <Headline
      textAlign="left"
      color="green"
      text="Programming is aboue expressing ideas" />

    <Headline
      textAlign="left"
      color="magenta"
      text="Languages should let us think, not inhibit us" />

    <Headline
      textAlign="left"
      color="blue"
      text="Idris" />

    <Points title="Idris">
      <Point text="Statically Typed" />
      <Point text="Functional" />
      <Point text="Pure" />
      <Point text="Dependently Typed" />
    </Points>

    <Code
      lang="haskell"
      source={`
        add1 : Int -> Int
        add1 x = x + 1
        
        greet : (name : String) -> String
        greet name = "Hello " ++ name

        not : Bool -> Bool
        not x = if true then false else true
      `}
    />

    <Headline
      textAlign="left"
      color="magenta"
      text="Types Restrict Values" />

    <Headline
      textAlign="left"
      color="green"
      text="Type Give Us Knowledge about Values" />

    <Code
      lang="haskell"
      source={`
        foo : Int -> Int
        foo x = ?foo
      `}
    />

    <Code
      lang="haskell"
      source={`
        bar : a -> a
        bar x = ?bar
      `}
    />

    <Code
      lang="haskell"
      source={`
        foo : Int -> Int -> Int
        foo x y = ?foo
      `}
    />

    <Code
      lang="haskell"
      source={`
        bar : a -> a -> a
        bar x y = ?bar
      `}
    />

    <Code
      lang="haskell"
      source={`
        foo : (f : Int -> Bool) -> (xs : List Int) -> List Bool
        foo f xs = ?foo
      `}
    />

    <Code
      lang="haskell"
      source={`
        foo : (f : Int -> Bool) -> (xs : List Int) -> List Bool
        foo f xs = map isEven xs
      `}
    />

    <Code
      lang="haskell"
      source={`
        bar : (f : a -> b) -> (xs : List a) -> List b
        bar f xs = ?bar
      `}
    />

    <Code
      lang="haskell"
      source={`
        bar : (f : a -> b) -> (xs : List a) -> List b
        bar f xs = []
      `}
    />

    <Headline
      textAlign="left"
      color="magenta"
      size={4}
      subtextSize={4}
      subtextCaps={true}
      text="Our Types are either too specific"
      subtext="or not specific enough" />

    <Code
      lang="haskell"
      source={`
        myVector : Vect 2 Int
        myVector = [1, 2]
      `}
    />

    <Code
      lang="haskell"
      source={`
        myVector : Vect 2 Int
        myVector = [1] -- Compile time error
      `}
    />

    <Code
      lang="haskell"
      source={`
        bar : (f : a -> b) -> (xs : Vect n a) -> Vect n b
        bar f xs = ?bar
      `}
    />

    <Headline
      textAlign="left"
      color="green"
      text="Demo" />

    <Points title="What we found">
      <Point text="Explain our code clearly" />
      <Point text="Use types wisely" />
      <Point text="Keep things pure" />
    </Points>

    <Headline
      textAlign="left"
      color="magenta"
      text="How does this work?" />

    <Points title="Restricted Search Space">
      <Point text="Function arguments" />
      <Point text="Recursion" />
      <Point text="Constructors" />
      <Point text="All type information known" />
    </Points>

    <Headline
      textAlign="left"
      color="blue"
      text="Barliman" />

    <Headline
      textAlign="left"
      color="green"
      text="TDD taken seriously" />

    <Headline
      textAlign="left"
      color="magenta"
      text="How does this work?" />







    

    <BlankSlide />

  </Presentation>
