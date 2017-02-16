
import React from "react";

import {
  Appear,
  BlockQuote,
  Cite,
  CodePane,
  Deck,
  Fill,
  Heading,
  Image,
  Layout,
  Link,
  ListItem,
  List,
  Markdown,
  Quote,
  Slide,
  Spectacle,
  Text
} from "@jimmyhmiller/spectacle";


import preloader from "@jimmyhmiller/spectacle/lib/utils/preloader";

const images = {
  me: require("./images/me.jpg"),
  functional: require("./images/functional.jpg"),
  haxl: require("./images/haxl.jpg"),
  propositions: require("./images/propositions.jpg"),
  wadler: require("./images/wadler.jpg"),
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
    const { noSlide=false, slide: Slide = Dark, ...props } = this.props;
    
    if (noSlide) {
      return <Comp {...props} />
    }
    return (
      <Slide>
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

const Code = withSlide(({ source, lang, title }) => {
  const spaces = detectIndent(source);
  return (
    <div>
      <Headline noSlide textAlign="left" text={title} />
      <CodePane textSize={20} source={removeIndent(spaces, source)} lang={lang} />
    </div>
  )
})
 
const Point = ({ text, textSize=50 }) => 
  <ListItem textSize={textSize} textColor="base2">
    {text}
  </ListItem>

const Points = withSlide(({ children, title, size, styleContainer }) =>
  <div style={styleContainer}>
    <Headline noSlide size={size} textAlign="left" text={title} />
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
    <Deck transition={["slide"]} transitionDuration={0} progress="none">
      {children}
    </Deck>
  </Spectacle>

export default () =>
  <Presentation>
    <Headline 
      size={2} 
      caps={false}
      subtextSize={3}
      text="Getting out of the Rut"
      subtext="A dive into Haskell" />

    <TwoColumn
      title="About Me"
      left={<Image src={images.me} />}
      right={
        <div style={{paddingTop: 80}}>
          <Text textColor="blue" textSize={60} textAlign="left">Jimmy Miller</Text>
          <Points noSlide styleContainer={{paddingTop: 10}}>
            <Point textSize={40} text="Self Taught" /> 
            <Point textSize={40} text="Lead Developer - Trabian" /> 
            <Point textSize={40} text="FP Nerd" />
          </Points>
        </div>
      } 
    />

    <Points title="Audience">
      <Point text="Experienced Developer" />
      <Point text="Bored" />
      <Point text="Open Minded" />
    </Points>

    <Headline 
      textAlign="left"
      text="haskell"
      subtext="An advanced, purely functional programming language" />

    <Points title="Purely Functional">
      <Point text="No Loops (for, while, etc)" />
      <Point text="No Variables" />
      <Point text="No Objects" />
      <Point text="No Side Effects" />
      <Point text="No Mutation" />
      <Point text="No Nulls" />
    </Points>

    <Headline caps={false} text="Haskell requires you to re-learn programming" />

    <Code 
      title="Hello World"
      lang="haskell"
      source={`
        main = putStr "Hello World!"
      `}
    />

    <Code 
      title="Functions"
      lang="haskell"
      source={`
        double x = x * 2
      `}
    />

    <Code 
      title="Functions"
      lang="haskell"
      source={`
        double :: Int -> Int
        double x = x * 2

        double 2 -- 4
      `}
    />

    <Code 
      title="Functions"
      lang="haskell"
      source={`
        add :: Int -> Int -> Int
        add x y = x + y

        add 2 3 -- 5
      `}
    />

    <Code 
      title="Functions"
      lang="haskell"
      source={`
        add :: Int -> Int -> Int
        add x y = x + y

        add2 :: Int -> int
        add2 = add 2

        add2 3 -- 5
      `}
    />

    <Code
      title="Currying" 
      lang="haskell"
      source={`
        add :: Int -> Int -> Int
          -- is the same as
        add :: Int -> (Int -> Int)
      `}
    />

    <Code
      title="Currying" 
      lang="haskell"
      source={`
        add 2 3
        -- is the same as
        (add 2) 3
      `}
    />

    <Code
      title="Currying" 
      lang="haskell"
      source={`
        addLots : Int -> Int -> Int -> Int
        ((addLots 2) 2) 2
        (addLots 2 2) 2
        addLots 2 2 2
      `}
    />

    <Code
      title="polymorphism"
      lang="haskell"
      source={`
        identity : Int -> Int
        identity x = x
      `}
    />

    <Code
      title="polymorphism"
      lang="haskell"
      source={`
        identity : a -> a
        identity x = x
      `}
    />

    <Code
      title="polymorphism"
      lang="haskell"
      source={`
        second : List a -> a
        second xs = first (rest xs)
      `}
    />

    <Code
      title="Pattern Matching" 
      lang="haskell"
      source={`
        fib :: Int -> Int
        fib 0 = 0
        fib 1 = 1
        fib n = fib (n - 1) + fib (n - 2)
      `}
    />

    <Code
      title="Pattern Matching" 
      lang="haskell"
      source={`
        not :: Bool -> Bool
        not True = False
        not False = True
      `}
    />

    <Code
      title="Data Types" 
      lang="haskell"
      source={`
        data Bool = True | False
      `}
    />

    <Code
      title="Data Types" 
      lang="haskell"
      source={`
        data Color = Green | Blue | Red
      `}
    />

    <Code
      title="Data Types"
      lang="haskell"
      source={`
        data Color = Green | Blue | Red

        isFavorite :: Color -> Bool
        isFavorite Green = True
        isFavorite _ = False
      `}
    />

    <Code
      title="Data Types"
      lang="haskell"
      source={`
        data Player 
          = Wizard Staff 
          | Warrior Sword
          | Ranger Bow

        data Sword = LongSword | ShortSword
        data Staff = IceStaff | FireStaff
        data Bow = LongBow | CrossBow

        myPlayer :: Player
        myPlayer = Warrior LongSword
      `}
    />

    <Code
      title="Data Types"
      lang="haskell"
      source={`
        data Special = Special Int | NotSpecial Int
      `}
    />

    <Code
      title="Data Types"
      lang="haskell"
      source={`
        data Special a = Special a | NotSpecial a
      `}
    />

    <Code
      title="No Nulls"
      lang="haskell"
      source={`
        data Maybe a = Nothing | Just a
      `}
    />
    <Code
      title="No Nulls"
      lang="haskell"
      source={`
        first :: List a -> Maybe a
        first [] = Nothing
        first (x : xs) = Just x
      `}
    />

    <Code
      title="No Nulls"
      lang="haskell"
      source={`
        data Player = Warrior (Maybe Sword)

        data Attack = Punch | Swing Sword

        attack :: Player -> Attack
        attack (Warrior Nothing) = Punch
        attack (Warrior (Just sword)) = Swing sword
      `}
    />

    <Code
      title="Error Handling"
      lang="haskell"
      source={`
        data Either a b = Left a | Right b
      `}
    />

    <Code
      title="Error Handling"
      lang="haskell"
      source={`
        data Item = DamagePotion | HealingPotion
        data Effect = Damage | Heal

        useItem :: Item -> Player -> Either String Effect
        useItem HealingPotion Wizard = Right Heal
        useItem DamagePotion Wizard = Right Damage
        useItem _ _ = Left "You can't use this item"
      `}
    />

    <Code
      title="Problem"
      lang="haskell"
      source={`
        helpTextPlayer :: Player -> String
        helpTextItem :: Item -> String
        helpTextSword:: Sword -> String
      `}
    />

    <Code
      title="Solution - Type Class"
      lang="haskell"
      source={`
        class Help a where
          helpText :: a -> String
      `}
    />

    <Code
      title="Type Classes"
      lang="haskell"
      source={`
        instance Help Staff where
          helpText IceStaff = "Uses Ice magic"
          helpText FireStaff = "Uses Fire magic"

        instance Help Player where
          helpText Warrior = "Strong!"
          helpText Wizard = "Magical"
          helpText Ranger = "Independent"

        helpText Warrior -- "Strong!"
        helpText IceStaff -- "Uses Ice magic"
      `}
    />

    <Code
      title="Type Classes"
      lang="haskell"
      source={`
        youShouldKnow :: Help a => a -> String
        youShouldKnow x = "You should know... " ++ helpText x
      `}
    />

    <Points title="Example Type Classes">
      <Point text="Eq (Equality)" />
      <Point text="Ord (Ordering)" />
      <Point text="ToJson" />
      <Point text="Num" />
      <Point text="Show" />
      <Point text="Functor" />
      <Point text="Alternative" />
      <Point text="Traversable" />
    </Points>

    <Code
      title="Do Notation"
      lang="haskell"
      source={`
        main = do
          putStrLn "Enter two words"
          word1 <- getLine
          word2 <- getLine
          putStr (word1 ++ " " ++ word2)
      `}
    />

    <Code
      title="Do Notation"
      lang="haskell"
      source={`
        getFoo :: Maybe Foo
        getBar :: Foo -> Maybe Bar
        getBaz :: Bar -> Maybe Baz

        getFooBarBaz :: Maybe (Foo, Bar, Baz)
        getFooBarBaz = do
          foo <- getFoo
          bar <- getBar foo
          baz <- getBaz bar
          return (foo, bar, baz)
      `}
    />

    <Headline text="What Haskell gives us?" />

    <Code
      title="Functional Programming"
      lang="haskell"
      source={`
        [1..100]
        |> map (+2)
        |> filter even
        |> filter (*2)
      `}
    />

    <Code
      title="Immutablilty"
      lang="haskell"
      source={`
        data Tree a 
          = Leaf a 
          | Branch (Tree a) a (Tree a)

        left :: Tree a -> a
        left (Leaf x) = x
        left (Branch l _ _) = left l
      `}
    />

    <Code
      title="Higher Level Abstractions"
      lang="haskell"
      source={`
        main = do
          cd "/tmp"
          mkdir "test"
          output "test/foo" "Hello, world!"
          stdout (input "test/foo")
          rm "test/foo"
          rmdir "test"
          sleep 1
          die "Urk!"
      `}
    />

    <Code
      title="Purity"
      lang="haskell"
      source={`
        moveTo :: Vec -> Ai ()
        moveTo pos = do
          arrived <- isAt pos
          unless arrived $ do
            angle <- angleTo pos
            rotateTowards angle
            accelerate
            moveTo pos
      `}
    />

    <Code
      title="Laziness"
      lang="haskell"
      source={`
        [0..]
        |> (*2)
        |> filter even
        |> take 10

        fibs = 0 : 1 : zipWith (+) fibs (tail fibs)
      `}
    />

    <ImageSlide src={images.functional} />

    <ImageSlide src={images.haxl} />

    <ImageSlide src={images.propositions} />

    <ImageSlide src={images.wadler} />

    <ImageSlide src={images.propositions} />

  </Presentation>
