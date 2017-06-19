
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

// Redux for the Perplexed

// Redux has risen to be the de facto standard for
// building large applications using React. Unfortunately 
// its popularity has led to a bandwagon effect of people 
// using Redux without understanding it. In this talk we 
// will discuss the why we might want to use Redux, how we 
// can use it, and finally discuss the benefits it brings 
// us.

// We will begin with a basic React app and discuss a 
// problem that occurs almost immediately when beginning 
// with React, where to store state. First we will 
// investigate some resolutions to this problems using 
// just vanilla React. Unfortunately, as our application 
// grows, these refactorings become more and more 
// burdensome and a better solution is needed. Redux was 
// made to fill this hole.

// Having motivated Redux, we will explore its three 
// fundamental concepts: reducers, actions, and the store. 
// We will modify our existing application to take 
// advantage of these features, alleviating the pains we 
// felt before and giving our code a strong organizing 
// principle. Finally, we will show what this refactor 
// enabled; using libraries from the redux ecosystem, we 
// will be able to quickly add undo/redo to our existing 
// code, persist our state to local storage, inspect our 
// program with a time travel debugger, and create truly 
// great bug reports with full playback of user actions.


export default () =>
  <Presentation>
    <Headline
      size={2}
      caps={false}
      subtextSize={3}
      text="Redux"
      subtext="for the perplexed" />

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

    <Headline
      text="Redux"
      subtext="for the perplexed" />

    <Headline
      color="blue"
      textAlign="left"
      text="Redux is a predictable state container for JavaScript apps." />

    <Headline
      color="red"
      text="The death of MVC" />




    <BlankSlide />

  </Presentation>
