
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
  fault: require("./images/fault.png"),
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

const Headline = withSlide(({ color="magenta", size=2, text, subtext, subtextSize, textAlign="center", caps=true, subtextColor, image }) =>
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
    {subtext && <Subtitle color={subtextColor} text={subtext} textAlign={textAlign} size={subtextSize} />}
    {image && <Image style={{paddingTop: 20}} height="70vh" src={image} />}
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
      size={2} 
      caps={false}
      subtextSize={3}
      text="Functional Programming"
      subtext="The Hard Sell" />

    <TwoColumn
      title="About Me"
      left={<Image src={images.me} />}
      right={
        <div style={{paddingTop: 80}}>
          <Text textColor="blue" textSize={60} textAlign="left">Jimmy Miller</Text>
          <Points noSlide styleContainer={{paddingTop: 10}}>
            <Point textSize={40} text="Self Taught" /> 
            <Point textSize={40} text="FP Nerd" />
          </Points>
        </div>
      } 
    />

    <Points title="What I won't do">
      <Point text="Give a definition of FP" />
      <Point text="Comprehsive Overview" />
      <Point text="Trash talk OOP" />
    </Points>

    <Headline
      color="green"
      text="Three Traps" />

    <Headline
      color="blue"
      text="1. Preaching to the choir" />

    <Headline
      color="red"
      text="2. Lecture" />

    <Headline
      color="yellow"
      text="3. Pie in the Sky" />


    <Points title="Plan of Action">
      <Point text="Motivate" />
      <Point text="Examples" />
      <Point text="Where are we going?" />
    </Points>

    <Headline
      color="green"
      text="The Hard Sell?" />

    <Points title="Motivations">
      <Point text="Positive" />
      <Point text="Negative" />
    </Points>

    <Headline
      color="blue"
      text="Get rid of bugs" />

    <Headline
      color="blue"
      text="A few bugs?" />

    <Headline
      color="blue"
      text="A whole class of bugs" />

    <Headline
      color="red"
      textAlign="left"
      text="I tried to click the button but nothing is working - Bob" />

    <Headline
      color="magenta"
      textAlign="left"
      text="How can this NOT work?" />

    <Headline
      color="blue"
      textAlign="left"
      text="How did this ever work?!?" />

    <Headline
      text="Who changed this value to that!?!?!?!" />

    <Headline
      color="green"
      text="Mutation causes bugs" />

    <Headline
      color="blue"
      text="Conjecture:"
      size={1}
      textAlign="left"
      subtext="All Heisenbugs are caused by mutation"
      subtextColor="green"
      subtextSize={2} />

    <Headline
      color="green"
      text="{Example Omitted}" />

    <Headline
      text="FP embraces immutability" />

    <Headline
      text="Other sorts of bugs?" />

    <Code
      title="No Nulls"
      lang="haskell"
      source={`
        data Maybe a = Nothing | Just a
      `}
    />

    <Points title="No Runtime Exceptions">
      <Point text="Written in Elm" />
      <Point text="50,000+ lines of code" />
      <Point text="Production for 2+ years" />
      <Point text="Zero runtime exceptions" />
    </Points>

    <Code
      title="Guarantee State Transitions"
      lang="haskell"
      source={`
        data Door = DoorOpen | DoorClosed

        data DoorCmd : Type -> Door -> Door -> Type where
             Open : DoorCmd () DoorClosed DoorOpen
             Close : DoorCmd () DoorOpen DoorClosed 
             Knock : DoorCmd () DoorClosed DoorClosed 

             ...
      `}
    />

    <Code
      title="Guarantee State Transitions"
      lang="haskell"
      source={`
        doorProg : DoorCmd () DoorClosed DoorClosed
        doorProg = do Knock
                      Open
                      Knock -- Doesn't compile
                      Close
        `}
      />

      <Headline
          text="Bugs Across Services"
          image={images.fault} />

      <Headline
        color="blue"
        text="Eliminate All the Bugs!!!" />

      <Headline 
        text="Positive" />

      <Headline 
        text="Expressivity" />

      <Headline
        text="Expressing our problem in code" />

      <Headline
        text="Kitchen Sink Approach"
        subtext="AKA the ruby approach" />

      <Headline
        text="Data Oriented Programming" />

      <Code
        title="Data Oriented Programming"
        lang="clojure"
        source={`
          ; html
          [:a {:href "#"} "Click Here"]

          ; sql
          (select user
            (with address)
            (fields :firstName :lastName :address.state)
            (where {:email "korma@sqlkorma.com"}))
        `}
      />

      <Code
        title="Data Oriented Programming"
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
        title="N+1 Fetches"
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
        title="Data Structures"
        lang="haskell"
        source={`
          data List a = Nil | Cons a (List a)
          data NEList a = NEList a (List a)
          data Tree a = Leaf a | Node (Tree a) a (Tree a)
          data Either a b = Left a | Right b
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

      <Headline
        text="Tooling" />

      <Headline
        color="blue"
        text="Truly Live Programming" />

      <Headline
        color="green"
        text="Type Driven Development" />

      <Headline
        text="A way forward" />

      <Headline
        color="blue"
        text="out of the tarpit" />

      <Points title="Future Directions">
        <Point text="Any Language" />
        <Point text="Academic or Practical" />
        <Point text="Lightning Talks" />
        <Point text="Help" />
      </Points>

    <BlankSlide />

  </Presentation>
