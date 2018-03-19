
import React from "react";

import {
  CodePane,
  Deck,
  Fill,
  Heading,
  Image,
  Layout,
  ListItem,
  List,
  Slide,
  Spectacle,
  Text
} from "@jimmyhmiller/spectacle";


import preloader from "@jimmyhmiller/spectacle/lib/utils/preloader";

const images = {
  me: require("./images/me.jpg"),
};

preloader(images);

// Import theme
import createTheme from "@jimmyhmiller/spectacle/lib/themes/default";


require("normalize.css");
require("@jimmyhmiller/spectacle/lib/themes/default/index.css");




const theme = createTheme({
  base03: "#002b36",
  base02: "#073642",
  base01: "#586e75",
  base00: "#657b83",
  base0: "#839496",
  base1: "#93a1a1",
  base2: "#eee8d5",
  base3: "#fdf6e3",
  yellow: "#b58900",
  orange: "#cb4b16",
  red: "#dc322f",
  magenta: "#d33682",
  violet: "#6c71c4",
  blue: "#268bd2",
  cyan: "#2aa198",
  green: "#859900",
});


// Spectacle needs a ref
class Dark extends React.Component {
  render() {
    const { children, ...rest } = this.props;
    return (
      <Slide bgColor="base03" {...rest}>
        {children}
      </Slide>
    )
  }
}

// Spectacle needs a ref
const withSlide = Comp => class WithSlide extends React.Component {
  render() {
    const { noSlide=false, slide: Slide = Dark, maxWidth, ...props } = this.props;
    
    if (noSlide) {
      return <Comp {...props} />
    }
    return (
      <Slide maxWidth={maxWidth}>
        <Comp {...props} />
      </Slide>
    )
  }
}

const detectIndent = source => 
  source ? / +/.exec(source)[0].length : 0;

const removeIndent = (indent, source) =>
  source.split("\n")
        .map(s => s.substring(indent, s.length))
        .join("\n")


const BlankSlide = withSlide(() => {
  return <span />;
})

const Code = withSlide(({ source, lang, title, textSize, headlineSize }) => {
  const spaces = detectIndent(source);
  return (
    <div>
      <Headline size={headlineSize} noSlide textAlign="left" text={title} />
      <CodePane textSize={textSize || 20} 
      source={removeIndent(spaces, source)} lang={lang} />
    </div>
  )
})
 
const Point = ({ text, textSize=50 }) => 
  <ListItem textSize={textSize} textColor="base2">
    {text}
  </ListItem>

const Points = withSlide(({ children, color, title, size, styleContainer }) =>
  <div style={styleContainer}>
    <Headline noSlide color={color} size={size} textAlign="left" text={title} />
    <List>
      {children}
    </List>
  </div>
)

const ImageSlide = withSlide(Image);

const Subtitle = ({ color="blue", size=5, text, ...props }) =>
  <Heading textColor={color} size={size} {...props}>
    {text}
  </Heading>

const Headline = withSlide(({ color="magenta", size=2, text, subtext, subtextSize, textAlign="center", caps=true }) =>
  <div>
    <Heading 
      textAlign={textAlign}
      size={size} 
      caps={caps} 
      lineHeight={1} 
      textColor={color}
    >
      {text}
    </Heading>
    {subtext && <Subtitle text={subtext} textAlign={textAlign} size={subtextSize} />}
  </div>
)

const TwoColumn = withSlide(({ left, right, title }) =>
  <div>
    <Layout>
      <Fill>
        <Headline color="cyan" size={4} textAlign="center" noSlide text={title} />
        {left}
      </Fill>
      <Fill>
        {right}
      </Fill>
    </Layout>
  </div>
)

const Presentation = ({ children }) => 
  <Spectacle theme={theme}>
    <Deck controls={false} style={{display: 'none'}} transition={["slide"]} transitionDuration={0} progress="none">
      {children}
    </Deck>
  </Spectacle>


export default () =>
  <Presentation>
    <Headline 
      text="What is a Monad?" />

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
      text="What is a Monad?" />

    <Headline
      color="blue"
      text="What is a thing?" />

    <Headline
      text="No Seriously" />

    <Points color="green" title="Being Lead Astray">
      <Point text="Concrete is Easier than Abstract" />
      <Point text="OOP makes us think in terms of things" />
      <Point text="Language is confusing" />
    </Points>

    <Points title="Two Ways of Viewing Things">
      <Point text="Concrete Defintions" />
      <Point text="Functional Abstractions" />
    </Points>

    <Points title="Singly Linked List">
      <Point text="Composed of Nodes and Pointers" />
      <Point text="Head of List Points to Tail" />
      <Point text="Has an Empty Node to signify end" />
    </Points>

    <Points title="Singly Linked List">
      <Point text="Supports head, tail, prepend, emptyList, isEmpty" />
      <Point text="head(prepend(x, coll)) == x"  />
      <Point text="tail(prepend(x, coll)) == coll" />
      <Point text="isEmpty(emptyList) == true" />
      <Point text="isEmpty(prepend(x, xs)) == false" />
    </Points>

    <Code
      title="Strange List"
      lang="javascript"
      source={`
        const emptyList = (selector) => 
          selector(null, null, true)

        const prepend = (elem, list) =>
          (selector) => selector(elem, list, false);

        const head = (list) => list((elem, list) => elem);

        const tail = (list) => list((elem, list) => list);

        const isEmpty = (list) => 
          list((elem, list, empty) => empty);

      `}
    />

    <Headline
      text="Monads are defined abstractly" />

    <Headline
      color="blue"
      textAlign="left"
      text="Monads are structures with certain operations and rules" />

    <Headline
      color="green"
      text="Why Care?" />

    <Headline
      text="Many Problems are Monadically Shapped" />

    <Headline
      color="blue"
      text="Monads Allow True Encapsulation" />

    <Code
      title="Lists"
      lang="javascript"
      source={`
        list = []
        for (i in coll1) {
          for (j in coll2) {
            for (k in coll3) {
              list.append(i + j +k)
            }
          }
        }
      `}
    />


    <Code
      title="Lists"
      lang="haskell"
      source={`
        do 
          i <- coll1
          j <- coll2
          k <- coll3
          return (i + j + k)
      `}
    />

    <Code
      title="Promises"
      lang="javascript"
      source={`
        promise1.then(x => 
          promise2.then(y =>
            promise3.then(z =>
               x + y + z
            )
          )
        )
      `}
    />


    <Code
      title="Promises"
      lang="haskell"
      source={`
        do 
          x <- promise1
          y <- promise2
          z <- promise3
          return (x + y + z)
      `}
    />

    <Code
      title="Null"
      lang="javascript"
      source={`
        if (foo !== null) {
          if (bar !== null) {
            if (baz != null) {
              return foo + bar + baz
            }
          }
        }
      `}
    />

    <Code
      title="Null"
      lang="haskell"
      source={`
        do 
          x <- foo
          y <- bar
          z <- baz
          return (x + y + z)
      `}
    />

    <Code
      title="Error Handling"
      lang="javascript"
      source={`
        try {
          const foo = getFoo();
          try {
            const bar = getBar();
            try {
              const baz = getBaz();
              return foo + bar + baz;
            }
            catch (e) {
              return "error3";
            }
          }
          catch (e) {
            return "error2";
          }
        } catch (e) {
          return "error1";
        }
      `}
    />

    <Code
      title="Error Handling"
      lang="haskell"
      source={`
        do 
          foo <- getFoo
          bar <- getBar
          baz <- getBaz
          return (foo + bar + baz)
      `}
    />

    <Code
        title="N+1 Fetches"
        lang="haskell"
        source={`
          getAllUsernames :: IO [Name]
          getAllUsernames = do
            userIds <- getAllUserIds
            for userIds $ \\userId -> do
              getUsernameById userId
        `}
      />

      <Code
        title="Not N+1 Fetches"
        lang="haskell"
        source={`
          getAllUsernames :: Haxl [Name]
          getAllUsernames = do
            userIds <- getAllUserIds
            for userIds $ \\userId -> do
              getUsernameById userId
        `}
      />

      <Code
        title="Parsing"
        lang="haskell"
        source={`
          threewords : Parser (List String)
          threewords = do
            word1 <- word
            spaces
            word2 <- word
            spaces
            word3 <- word
            pure [word1, word2, word3]
        `}
      />

      <Code
        title="Any Program"
        lang="haskell"
        source={`
          moveTo :: Vector -> Ai ()
          moveTo position = do
            arrived <- isAt position
            unless arrived $ do
              angle <- angleTo position
              rotateTowards angle
              accelerate
              moveTo position
        `}
      />

      <Headline
        text="Deriving Monads" />

      <Code
        lang="haskell"
        source={`
          > result = map isEven [1,2,3,4]
          > result
          [false, true, false, true]
        `}
      />

      <Code
        lang="haskell"
        source={`
          result = map isEven [1,2,3,4]
          
          isEven : Int -> Bool
          [1,2,3,4] : List Int
          result : List Bool

          map : (Int -> Bool) -> List Int -> List Bool
        `}
      />

      <Code
        lang="haskell"
        source={`
          result = map length ["a", "ab", "abc"]
          > result
          [1, 2, 3]
        `}
      />

      <Code
        lang="haskell"
        source={`
          result = map length ["a", "ab", "abc"]
          
          length : String -> Int
          ["a", "ab", "abc"] : List String
          result : List Int

          map : (String -> Int) -> List String -> List Int
        `}
      />

      <Code
        lang="haskell"
        source={`
          map : (Int -> Bool)   -> List Int    -> List Bool
          map : (String -> Int) -> List String -> List Int
        `}
      />

      <Code
        maxWidth={1200}
        lang="haskell"
        source={`
          map : (Int    -> Bool)   -> List Int    -> List Bool
          map : (String -> Int)    -> List String -> List Int
         
          map : (a      -> b)      -> List a      -> List b
        `}
      />

      <Code
        lang="haskell"
        source={`
          map : (a -> b) -> List a -> List b
        `}
      />

      <Code
        lang="haskell"
        source={`
          > map (+2) [1]
          [3]

          > map (+2) (Just 1)
          Just 3

          > map (+2) (Id 1)
          Id 3
        `}
      />

      <Code
        lang="haskell"
        source={`
          map : (a -> b) -> List a -> List b
          map (+2) [1]
          
          map : (a -> b) -> Maybe a -> Maybe b
          map (+2) (Just 1)

          map : (a -> b) -> Identity a -> Identity b
          map (+2) (Id 1)
        `}
      />

      <Code
        lang="haskell"
        source={`
          map : (a -> b) -> List a     -> List b
          map : (a -> b) -> Maybe a    -> Maybe b
          map : (a -> b) -> Identity a -> Identity b

          map : (a -> b) -> f a        -> f b
        `}
      />

      <Code
        lang="haskell"
        source={`
          map : (a -> b) -> f a -> f b
        `}
      />

      <Code
        lang="haskell"
        source={`
          interface Functor f where
            map : (a -> b) -> f a -> f b
        `}
      />


      <Code
        lang="haskell"
        source={`
          interface Functor f => Applicative f where
            pure : a -> f a
            ap : f (a -> b) -> f a -> f b
        `}
      />

      <Code
        lang="haskell"
        source={`
          interface Applicative f => Monad f where
            bind : f a -> (a -> f b) -> f b
        `}
      />










    <BlankSlide />

  </Presentation>
