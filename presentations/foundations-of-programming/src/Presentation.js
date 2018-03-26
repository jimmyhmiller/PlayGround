
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
      text="The Foundations of Programming" />

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
      text="The Foundations of Programming" />

    <Headline
      color="blue"
      text="Live Code" />


    <Headline
      size={4}
      textAlign="left"
      color="green"
      text="Functions are the most fundamental feature of programming" />

    <Points title="Lambda Calculus">
      <Point text="Simple Definition" />
      <Point text="e ::= x | λx.e | e e" />
      <Point text="Turing Complete" />
    </Points>

    <Points size={4} title="Simply Typed Lambda Calculus">
      <Point text="Simple Definition" />
      <Point text="e ::= x | λx:τ.e  | e e | c" />
      <Point text="Not Turing Complete" />
    </Points>

    <Points title="Predates Programming">
      <Point text="Discovered in 1932" />
      <Point text="By Alonzo Church" />
      <Point text='"A set of postulates for the foundation of logic"' />
    </Points>

    <Points title="Scope of Lambda Calculus">
      <Point text="Involves the Foundations of Math" />
      <Point text="Isomorphic to Logic" />
      <Point text="For every logic, there is a computation" />
    </Points>

    <Points title="Logics">
      <Point text="Intuitionistic Logic = Programming" />
      <Point text="Modal Logic = Distributed Programming" />
      <Point text="Temporal Logic = Reactive Programming" />
      <Point text="Linear Logic = Memory Safe Programming" />
    </Points>

    <Points title="Programming is Applied Logic">
      <Point text="Types are Propositions" />
      <Point text="Programs are Proofs" />
      <Point text="Program Execution is Proof Simplification" />
    </Points>

    <Headline
      text="Programming is fundamental" />

    <Headline
      text="Programming is beautiful" />










    <BlankSlide />

  </Presentation>
