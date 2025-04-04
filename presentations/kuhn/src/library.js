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
  BlockQuote,
  Quote,
} from "spectacle";
import pre from "spectacle/lib/utils/preloader";
import createTheme from "spectacle/lib/themes/default";



export {
  CodePane,
  Deck,
  Fill,
  Heading,
  Image,
  Layout,
  ListItem,
  List,
  Slide,
  Text,
} from "spectacle";


export const preloader = pre;

// Import theme



require("normalize.css");




export const theme = createTheme({
  base03: "#002b36",
  base02: "#073642",
  base01: "#586e75",
  base00: "#657b83",
  base0: "#839496",
  base1: "#93a1a1",
  base2: "#eee8d5",
  base3: "#fdf6e3",
  yellow: "#d4a701",
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
export const withSlide = Comp => class WithSlide extends React.Component {
  render() {
    const { noSlide=false, slide: Slide = Dark, maxWidth, maxHeight, ...props } = this.props;
    
    if (noSlide) {
      return <Comp {...props} />
    }
    return (
      <Slide {...props} maxWidth={maxWidth || 1300} maxHeight={maxHeight || 800}>
        <Comp {...props} />
      </Slide>
    )
  }
}

const removeFirst = (arr) => {
  arr.shift();
  return arr
}

export const detectIndent = source => 
  source ? / +/.exec(source)[0].length : 0;

export const removeIndent = (indent, source) =>
  removeFirst(source.split("\n"))
        .map(s => s.substring(indent, s.length))
        .join("\n")


export const formatCode = (source) => {
  const spaces = detectIndent(source);
  return removeIndent(spaces, source)
}

export const BlankSlide = withSlide(() => {
  return <span />;
})

export const Code = withSlide(({ source, color, lang, title, textSize, headlineSize }) => {
  return (
    <div>
      {title ? <Headline color={color} size={headlineSize || 4} noSlide textAlign="left" text={title} /> : null}
      <CodePane
        theme="external"
        textSize={textSize || 36} 
        source={formatCode(source)}
        lang={lang} />
    </div>
  )
})
 
export const Point = ({ text, textSize=50 }) => 
  <ListItem textSize={textSize} textColor="base2">
    {text}
  </ListItem>

export const Points = withSlide(({ children, color, title, size=3, styleContainer, caps }) =>
  <div style={styleContainer}>
    <Headline noSlide caps={caps} color={color} size={size} textAlign="left" text={title} />
    <List>
      {children}
    </List>
  </div>
)

export const QuoteSlide = withSlide(({ text, color="yellow" }) =>
  <BlockQuote>
    <Quote textColor={color}>{text}</Quote>
  </BlockQuote>

)

export const ImageSlide = ({Slide=Dark, src, height, maxWidth, maxHeight, caps, color, size=3, title, align, ...props }) => 
  <Slide maxWidth={maxWidth || 1300} maxHeight={maxHeight || 800} {...props}>
    <Headline noSlide caps={caps} color={color} size={size} textAlign={ align || "left" } text={title} />
    <Image style={{paddingTop: 30}} src={src} height={height} />
  </Slide>

export const Subtitle = ({ color="blue", caps, size=5, text, ...props }) =>
  <Heading caps={caps} textColor={color} size={size} {...props}>
    {text}
  </Heading>

const headlineColors = [
  "yellow",
  "red",
  "magenta",
  "blue",
  "cyan",
  "green",
]


export const Headline = withSlide(({ color, size=2, text, subtext, subtextSize, subtextCaps, textAlign="center", caps=true, slideIndex }) =>
  <div>
    <Heading 
      textAlign={textAlign}
      size={size} 
      caps={caps} 
      lineHeight={1} 
      textColor={color || slideIndex ? headlineColors[slideIndex % headlineColors.length] : "magenta"}
    >
      {text}
    </Heading>
    {subtext && <Subtitle caps={subtextCaps} text={subtext} textAlign={textAlign} size={subtextSize} />}
  </div>
)

export const TwoColumn = withSlide(({ left, right, title, align="left", color="cyan", size=2 }) =>
  <div style={{paddingTop: 80}}>
    <Headline color={color} size={size} textAlign={align} noSlide text={title} />
    { title && <div style={{paddingTop: 80}} />}
    <Layout>
      <Fill>
        {left}
      </Fill>
      <Fill>
        {right}
      </Fill>
    </Layout>
  </div>
)

export const Presentation = ({ children }) => 
  <Deck contentWidth={1400} contentHeight={950} theme={theme} controls={false} style={{display: 'none'}} transition={["slide"]} transitionDuration={0} progress="none">
    {children}
  </Deck>
