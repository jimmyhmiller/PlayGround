
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
      text="Datalog"
      subtext="A Functional Alternative to SQL" />

    <Headline
      text="New Title" />

    <Headline 
      size={1} 
      caps={false}
      subtextSize={4}
      textAlign="left"
      text="Datomic"
      subtext="Best DB Ever" />

    <Headline
      text="SQL makes me angry" />

    <Headline
      color="blue"
      text="ORMs make me angier" />

    <Headline
      color="green"
      text="90% of tech debt comes from bad data modeling" />

    <Points title="A Better Way">
      <Point text="No More Tables" />
      <Point text="Better joins" />
      <Point text="Make Time Travel Possible" />
    </Points>

    <Code
      lang="clojure"
      title="Uniform Data Model"
      source={`
        [entity attribute value]
      `}
    />

    <Code
      lang="clojure"
      title="Uniform Data Model"
      source={`
        [1 :person/name "Jimmy"]
        [1 :person/age 26]

        [2 :person/name "Bob"]
        [2 :person/age 28]
      `}
    />

    <Code
      lang="clojure"
      headlineSize={4}
      title="Pattern Matching Queries"
      source={`
        {:find [?name]
         :where [[?e :person/name ?name]]}
      `}
    />

    <Code
      lang="clojure"
      headlineSize={4}
      title="Pattern Matching Queries"
      source={`
        {:find [?name ?age]
         :where [[?e :person/name ?name]
                 [?e :person/age ?age]]}
      `}
    />

    <Code
      lang="clojure"
      headlineSize={4}
      title="All People With Same Age"
      source={`
        {:find [?e1 ?e2]
         :where [[?e1 :person/age ?age]
                 [?e2 :person/age ?age]]}
      `}
    />

    <Code
      lang="clojure"
      headlineSize={4}
      title="All attributes with value"
      source={`
        {:find [?e ?attr]
         :where [[?e ?attr 26]]}
      `}
    />



    <Code
      lang="clojure"
      headlineSize={4}
      title="Put data in the shape you need"
      source={`
        (pull db
          [:person/first-name
           :person/last-name
           {:person/address [:address/zipcode]}])

        {:person/first-name "Jimmy"
         :person/last-name "Miller"
         :person/address {:address/zipcode "46203"}}
      `}
    />

    <Code
      maxWidth={1200}
      lang="clojure"
      headlineSize={4}
      title="Put data in the shape you need"
      source={`
        (q '{:find [(pull ?e [:person/first-name
                              :person/last-name
                              {:person/address [:address/zipcode]}]) ...]
             :in [$ ?zip]
             :where [[?a :address/zipcode ?zip]
                     [?e :person/address ?a]]}
           db
           "46203")
      `}
    />


    <Code
      lang="clojure"
      headlineSize={4}
      title="Get data when you need it"
      source={`
        (q '{:find [?name]
             :in [$]
             :where [[?e :person/name ?name]]}
           db)
      `}
    />

    <Code
      lang="clojure"
      headlineSize={4}
      title="Data at a time in the past"
      source={`
        (q '{:find [?name]
             :in [$]
             :where [[?e :person/name ?name]]}
           (as-of db yesterday))
      `}
    />


    <Code
      lang="clojure"
      headlineSize={4}
      title="Data across all time"
      source={`
        (q '{:find [?name]
             :in [$]
             :where [[?e :person/name ?name]]}
           (history db))
      `}
    />

    <Code
      lang="clojure"
      headlineSize={4}
      title="Data in the future"
      source={`
        (q '{:find [?name]
             :in [$]
             :where [[?e :person/name ?name]]}
           (with db changes))
      `}
    />

    <Points title="Even More">
      <Point text="Transaction Metadata" />
      <Point text="Schema as Data" />
      <Point text="Subscribe to Transactions" />
      <Point text="Recursive Queries" />
      <Point text="Compositional Queries" />
      <Point text="Query Multiple DBs" />
    </Points>

    <BlankSlide />


  </Presentation>
