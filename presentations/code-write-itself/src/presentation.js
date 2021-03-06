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

    <Points title="Why?">
      <Point text="Writing Code is error prone" />
      <Point text="Waste of time" />
      <Point text="This is what programming is about" />
    </Points>

    <Headline
      textAlign="left"
      color="green"
      text="Programming is about expressing ideas" />

    <Points title="Approach">
      <Point text="Idris - Static" />
      <Point text="Barliman - Dynamic" />
      <Point text="Both experimental" />
    </Points>

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
        not x = if x then false else true
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
        foo x = _
      `}
    />

    <Code
      lang="haskell"
      source={`
        bar : a -> a
        bar x = _
      `}
    />

    <Code
      lang="haskell"
      source={`
        foo : (f : Int -> Bool) -> (xs : List Int) -> List Bool
        foo f xs = _
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
        bar f xs = _
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
        bar f xs = _
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
      color="blue"
      size={1}
      text="Barliman" />

    <Headline
      textAlign="left"
      color="green"
      text="TDD taken seriously" />

    <Headline
      textAlign="left"
      color="magenta"
      text="How does this work?" />

    <Headline
      textAlign="left"
      color="blue"
      caps={false}
      text="miniKanren" />

    <Points title="miniKanren" caps={false}>
      <Point text="Embedded language" />
      <Point text="Logic Programming Language" />
      <Point text="Weird" />
      <Point text="Has Magic Powers" />
    </Points>

    <Code
      lang="clojure"
      source={`
      (run 1 (q)
           (== q 1))
      ;; => 1
      `}
    />

    <Code
      lang="clojure"
      source={`
        (run 1 (q)
             (conde
              [(== q 1) succeed]))
        ;; => 1
      `}
    />

    <Code
      lang="clojure"
      source={`
        (run* (q)
             (conde
              [(== q 1) succeed]
              [(== q 2) succeed]))
        ;; => (1 2)
      `}
    />

    <Code
      lang="clojure"
      source={`
        (run 1 (q)
             (appendo '(1 2 3) '(4 5 6) q))
        ;; => (1 2 3 4 5 6)
      `}
    />

    <Code
      lang="clojure"
      source={`
        (run 1 (q)
             (appendo '(1 2 3) q '(1 2 3 4 5 6)))
        ;; => (4 5 6)
      `}
    />

    <Code
      lang="clojure"
      source={`
        (run 1 (q)
             (appendo q '(4 5 6) '(1 2 3 4 5 6)))
        ;; => (1 2 3)
      `}
    />

    <Code
      lang="clojure"
      source={`
        (define numbers
          (lambda (x q)
            (conde
             [(== x 1) (== q "one")]
             [(== x 2) (== q "two")]
             [(== x 3) (== q ">2")]
             [(== x 4) (== q ">2")])))
      `}
    />

    <Code
      lang="clojure"
      source={`
        (run* (q)
             (numbers 4 q))
        ;; => (">2")

        (run* (q)
             (numbers q "one"))
        ;; => (1)
      `}
    />

    <Code
      lang="clojure"
      source={`
        (run* (q)
             (numbers q ">2"))
        ;; => (3 4)

        (run* (q r)
             (numbers q r))

        ;; => (1 2 3 4)
      `}
    />

    <Headline
      textAlign="left"
      color="magenta"
      text="What if we wrote an interpreter?" />

    <Code
      lang="clojure"
      source={`

        (run 1 (q)
             (evalo \`(if (equal? #t #t) 1 2) q))
        ;; => 1

        (run 1 (q)
             (evalo \`(if (equal? ,q #t) 1 2) 1))
        ;; => #t
      `}
    />


    <Code
      lang="clojure"
      source={`

        (run 4 (q)
             (evalo \`(if (equal? ,q #t) 1 2) 1))
        
        ;; => (#t ((lambda _ #t)) (and) (not #f))
      `}
    />

    <Code
      lang="clojure"
      source={`

        (run 4 (q)
             (evalo q q))
        
      `}
    />


    <Code
      lang="clojure"
      source={`

        (run 4 (q)
             (evalo q q))
        
        ;; (num _)
        ;; #t
        ;; #f
        ;; ((lambda (_) (list _ (list 'quote _)))
        ;;   '(lambda (_) (list _ (list 'quote _))))

      `}
    />


    <Code
      lang="clojure"
      source={`

        > ((lambda (_) (list _ (list 'quote _)))
            '(lambda (_) (list _ (list 'quote _))))

        ((lambda (_) (list _ (list 'quote _)))
          '(lambda (_) (list _ (list 'quote _))))

      `}
    />

    <Points title="How it works">
      <Point text="Lisp interpreter in miniKanren" />
      <Point text="Smart search for potential code" />
      <Point text="Runs tests to check" />
    </Points>


    <Points title="What have we seen?">
      <Point text="Experimental, not ready" />
      <Point text="Each largely done by a single person" />
      <Point text="Provides new opportunities to learn" />
    </Points>


    <Headline
      textAlign="left"
      color="blue"
      text="Programming is in its infancy" />


    <BlankSlide />

  </Presentation>
