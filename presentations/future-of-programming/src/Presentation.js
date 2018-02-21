
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
    <Deck controls={false} style={{display: 'none'}} transition={["slide"]} transitionDuration={0} progress="none">
      {children}
    </Deck>
  </Spectacle>


export default () => (
  <Presentation>
    <Headline 
      size={2} 
      caps={false}
      subtextSize={3}
      text="The Future of Programming" />

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
      size={2} 
      caps={false}
      subtextSize={3}
      text="The Future of Programming" />
    
    <Points title="Limited Scope">
      <Point text="Focus on three languages" />
      <Point text="Aimed at near future" />
      <Point text="Broad overview" />
    </Points>

    <Headline 
      color="green"
      text="The Sorry State of Programming" />

    <Points title="Problems">
      <Point text="Our programs are full of bugs" />
      <Point text="Artifical boundries between machines" />
      <Point text="It takes way too much code to accomplish anything" />
    </Points>

    <Headline 
      color="blue"
      text="Getting out of the Tarpit" />

    <Points title="Difficulties in moving forward">
      <Point text="We are fickle" />
      <Point text="We are afraid to learn" />
      <Point text="We are obsessed with easiness" />
    </Points>

    <Headline
      text="Idris" />

    <Headline
      text="What if types were first class?" />

    <Headline
      color="green"
      text="What if our programs were proofs?" />

    <Points title="How to get here?">
      <Point text="Get rid of OOP" />
      <Point text="Be rigorous" />
      <Point text="Consider our assumptions" />
      <Point text="Learn some math" />
    </Points>

    <Points title="Get a taste today">
      <Point text="Type Driven Development" />
      <Point text="Write some Haskell" />
    </Points>

    <Headline
      text="Unison" />

    <Headline
      text="What if other machines were first class?" />

    <Headline
      color="blue"
      text="What if our code was immutable?" />

    <Code
      title="Hello Unison"
      lang="haskell"
      source={`
        -- alice : Node, bob : Node

        do Remote
          x = factorial 6
          Remote.transfer alice
          y = foo x -- happens on 'alice' node
          Remote.transfer bob
          pure (bar x y) -- happens on 'bob' node
      `} 
    />

    <Headline
      text="Eve" />
    
    <Headline
      color="blue"
      size={4}
      text="What if we were immediately connected to our creations?" />

    <Headline
      text="What if our code was distributed by default?" />

    <Headline
      color="green"
      text="What if it otook an order of magnitude less code?" />

    <Headline
      text="What if anyone could program?" />

    <BlankSlide />

  </Presentation>
)
