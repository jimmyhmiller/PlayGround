
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


const BlankSlide = withSlide(() => {
  return <span />;
})

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

    <Headline text="Two Ruts" color="green" />

    <BlankSlide />

    <Points title="Curry Howard Isomorphism">
      <Point text="Types are Propositions" />
      <Point text="Programs are Proofs" />
      <Point text="Program Execution is Proof Simplification" />
    </Points>

    <Headline
     text="Haskell Embraces Mathematics"
     subtext="and so should we" />

    <Points title="Purely Functional">
      <Point text="No Loops (for, while, etc)" />
      <Point text="No Variables" />
      <Point text="No Objects" />
      <Point text="No Side Effects" />
      <Point text="No Mutation" />
      <Point text="No Nulls" />
    </Points>

    <BlankSlide />

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

        double 2 
        -- 4
      `}
    />

    <Code 
      title="Functions"
      lang="haskell"
      source={`
        add :: Int -> Int -> Int
        add x y = x + y

        add 2 3 
        -- 5
      `}
    />

    <Code 
      title="Functions"
      lang="haskell"
      source={`
        greet :: String -> String
        greet name = "Hello " ++ name

        greet "Bob" 
        -- "Hello Bob"
      `}
    />

    <BlankSlide />

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

    <BlankSlide />

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
      title="Pattern Matching" 
      lang="haskell"
      source={`
        isFavorite :: Color -> Bool
        isFavorite Green = True
        isFavorite _ = False
      `}
    />

    <Code
      title="Pattern Matching" 
      lang="haskell"
      source={`
        attackMultiplier :: Player -> Float
        attackMultiplier (Warrior LongSword) = 3
        attackMultiplier (Warrior ShortSword) = 2
        attackMultiplier (Ranger _) = 0.8
        attackMultiplier (Wizard _) = 0.5
      `}
    />

    <BlankSlide />

    <Code
      title="polymorphism"
      lang="haskell"
      source={`
        identityInt : Int -> Int
        identityInt x = x

        identityString : String -> String
        identityString x = x
      `}
    />

    <Code
      title="polymorphism"
      lang="haskell"
      source={`
        identity : a -> a
        identity x = x

        identity "Test"
        identity False
        identity 1
      `}
    />

    <Code
      title="Data Types"
      lang="haskell"
      source={`
        data IsSpecialInt 
          = Special Int 
          | NotSpecial Int

        data IsSpecialString 
          = Special String 
          | NotSpecial String
      `}
    />

    <Code
      title="Data Types"
      lang="haskell"
      source={`
        data IsSpecial a = Special a | NotSpecial a
      `}
    />


    <Code
      title="Data Types"
      lang="haskell"
      source={`
        data IsSpecial a = Special a | NotSpecial a

        special :: IsSpecial String
        special = Special "Everyone"
      `}
    />

    <BlankSlide />

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
      title="No Nulls"
      lang="haskell"
      source={`
        getFoo :: Maybe Foo
        getBar :: Foo -> Maybe Bar
        getBaz :: Bar -> Maybe Baz

        fooBarBaz :: Maybe (Foo, Bar, Baz)
        fooBarBaz = do
          foo <- getFoo
          bar <- getBar foo
          baz <- getBaz bar
          return $ (foo, bar, baz)
      `}
    />

    <BlankSlide />

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
      title="Error Handling"
      lang="haskell"
      source={`
        merge :: Data -> Data -> Data
        getData :: Id -> Either StoreError Data
        storeData :: Data -> Either StoreError Id

        updateData :: Id -> Data -> Either StoreError Id
        updateData id currentData = do
          oldData <- getData isRequired
          let newData = merge oldData currentData
          newId <- storeData newData
          return newId


      `}
    />

    <BlankSlide />

    <Code
      title="Problem"
      lang="haskell"
      source={`
        helpTextPlayer :: Player -> String
        helpTextItem :: Item -> String
        helpTextSword :: Sword -> String
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
      <Point text="Monad" />
      <Point text="Alternative" />
      <Point text="Traversable" />
    </Points>

    <BlankSlide />

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

    <Headline text="Right Reasoning" />

    <ImageSlide src={images.haxl} />

  </Presentation>
