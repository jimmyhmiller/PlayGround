
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


// # Interpreter

// 1. Numbers
// 2. Booleans
// 3. Strings
// 4. Addition
// 5. Zero?
// 6. If
// 7. Let
// 8. Function Application
// 9. Function Declaration
// 10. Globals
// 11. Hello World
// 12. Interop
// 13. Call-by-need


export default () =>
  <Presentation>
    <Headline 
      size={2} 
      caps={false}
      subtextSize={2}
      textAlign="left"
      text="Clojure"
      subtext="A Programming Superpower" />

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
      subtextSize={2}
      textAlign="left"
      text="Clojure"
      subtext="A Programming Superpower" />

    <Headline 
      color="green"
      size={1}
      text="We are obsessed with ease" />

    <Headline
      color="blue"
      text="Focus on ease causes productivity decrease" />

    <Headline
      text="Productivity in the Long Term Matters" />

    <Headline
      color="cyan"
      text="Only Simplicity Scales" />

    <Points title="Clojure - The Highlights" size={4}>
      <Point text="Immutable" />
      <Point text="Functional" />
      <Point text="Practical" />
      <Point text="Hosted" />
    </Points>

    <Code
      title="Intro To Clojure"
      lang="clojure"
      source={`
        1 ; Integer
        1.0 ; Double
        true ; Boolean
        "thing" ; String
        :name ; Keyword
        x ; Symbol
      `} />


    <Code
      title="Intro To Clojure"
      lang="clojure"
        source={`
        [1 2 3] ; Vector
        (1 2 3) ; List
        {:a 3 :b "adf"} ; Map
        #{1 2 3} ; Set
      `} />

    <Code
      title="Code is Data"
      lang="clojure"
      source={`
        (defn add [x y]
          (+ x y))

        (add 3 5)
      `} />

    <Code
      title="Code is Data"
      lang="clojure"
      source={`
        (->>
         (for
           [x (range 100 1000)
            y (range 100 1000)
              :while (< y x)]
           (* x y))
         (filter palindrome?)
         (apply max))

      `} />



    <BlankSlide />

  </Presentation>
