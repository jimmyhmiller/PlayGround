
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

import {format as prettier} from 'prettier';


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
  primary: "#002b36",
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

// // Spectacle needs a ref
// class Light extends React.Component {
//   render() {
//     const { children, ...rest } = this.props;
//     return (
//       <Slide bgColor="base3" {...rest}>
//         {children}
//       </Slide>
//     )
//   }
// }


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
  source
    .replace(/^\n/, '')
    .split("\n")
    .map(s => s.substring(indent, s.length))
    .join("\n")

const formatSource = ({lang, source}) => {
  if (lang === 'javascript' || lang === "jsx") {
    return prettier(source, { printWidth: 50 });
  }
  return source;
}

const Code = withSlide(({ source, lang, title, titleSize }) => {
  const spaces = detectIndent(source);
  const unindentedSource = removeIndent(spaces, source);
  const formattedSource = formatSource({ lang, source: unindentedSource });
  return (
    <div>
      <Headline size={titleSize} noSlide textAlign="left" text={title} />
      <CodePane textSize={20} source={formattedSource} lang={lang} />
    </div>
  )
})
 
const Point = ({ text, textSize=50 }) => 
  <ListItem textSize={textSize} textColor="base2">
    {text}
  </ListItem>

const Points = withSlide(({ children, title, size, styleContainer, color }) =>
  <div style={styleContainer}>
    <Headline color={color} noSlide size={size} textAlign="left" text={title} />
    <List>
      {children}
    </List>
  </div>
)

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
        {title && <Headline color="cyan" size={4} textAlign="center" noSlide text={title} />}
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
    <Deck controls={false} transition={["slide"]} transitionDuration={0} progress="none">
      {children}
    </Deck>
  </Spectacle>

export default () =>
  <Presentation>
    <Headline 
      size={2}
      caps={false}
      subtextSize={3}
      color="blue"
      text={<span>Clojure<span style={{color: "#2aa198"}}>(Script)</span> for Javascript Developers</span>} />

    <Points title="What is Clojure(Script)">
      <Point text="Functional Language" />
      <Point text="Lisp" />
      <Point text="JVM/Javascript" />
    </Points>

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

    <Headline text="Javascript Fatigue" />

    <Points title="Fatigue Causers">
      <Point text="Build Tools" />
      <Point text="ES6/ES7/ES2015/ES2016/ES.Next" />
      <Point text="So many libraries" />
    </Points>


    <Points color="blue" title="Progress">
      <Point text="Better Tools" />
      <Point text="New Language Features" />
      <Point text="Functional Programming" />
    </Points>

    <Headline color="green" text="Going Against the Grain" />

    <Code
      lang="javascript"
      source={`
        function updateVeryNestedField(state, action) {
            return {
                ...state,
                first : {
                    ...state.first,
                    second : {
                        ...state.first.second,
                        [action.someId] : {
                            ...state.first.second[action.someId],
                            fourth : action.someValue
                        }
                    }
                }
            }
        }
      `}
    />

    <Points title="Functional AltJS">
      <Point text="Elm" />
      <Point text="Purescript" />
      <Point text="Reason" />
    </Points>

    <Headline
      color="blue"
      text="What Sets Clojure Apart?" />

    <Points title="Compared to Javascript">
      <Point text='No "wats"' />
      <Point text="Functional by Design" />
    </Points>

    <Points title="Compared to Alternatives">
      <Point text="Dynamically Typed" />
      <Point text="Mature/Production Ready" />
    </Points>


    <Headline 
      color="cyan"
      text="Compared to All?" />

    <Headline 
      color="red"
      text="Flexibility to Add Features to the Language" />

    <Points color="green" title="JS Language Features">
      <Point text="ES.Next" />
      <Point text="JSX" />
      <Point text="TypeScript/Flow" />
    </Points>

    <Points color="green" title="AltJS Language Features">
      <Point text="Pattern Matching" />
      <Point text="Static Typing" />
    </Points>

    <Headline
      color="blue"
      text="Write a compiler" />

    <Headline
      color="green"
      text="The Democratization of Programming" />

    <Headline
      caps={false}
      text="(> semantics syntax)" />

    <Code
      title="Immutable Datastructures"
      lang="clojure"
      source={`
        ;; list
        (1 2 3)

        ;; vector
        [1 2 3]

        ;; map
        {:name "jimmy"
         :favorite-food "ice cream"}
      `}
    />

    <Code
      title="(= data-structures syntax)"
      titleSize={4}
      lang="clojure"
      source={`
        (defn double [x]
          (* x 2))

        (double 2) ;; 4
      `}
    />

    <Code
      title="(= data-structures syntax)"
      titleSize={4}
      lang="clojure"
      source={`
        (def x 3)
        
        (if (= x 2)
          "It's two"
          "It's not two")
      `}
    />

    <Code
      title="(= data-structures syntax)"
      titleSize={4}
      lang="clojure"
      source={`
        (->> (range 100)
             (map (partial + 2))
             (filter even?)
             (reduce + 0))
      `}
    />

    <Code
      title="(= data-structures syntax)"
      titleSize={4}
      lang="clojure"
      source={`
        (defn update-very-nested-field [state action]
          (assoc-in state [:first :second 
                           (:some-id action) :forth] 
                    (:some-value action)))
      `}
    />

    <Headline text="((Parentheses) are Scary!!!)" />

    <Code
      lang="javascript"
      title="JSX"
      source={`
        const HomeLink = () => {
         return <a href="#">Home</a>
        }
      `}
    />

    <Code
      lang="clojure"
      title="Plain Old Clojure"
      source={`
        (defn home-link []
          [:a {:href "#"} "Home"])
      `}
    />

    <Code
      lang="javascript"
      source={`
        function getUserAndAddress(id) {
          const user = getUser(id);
          const address = getAddress(user.addressId);
          return merge(user, address);
        }
      `}
    />

    <Code
      lang="javascript"
      title="Async Await"
      source={`
        async function getUserAndAddress(id) {
          const user = await getUser(id);
          const address = await getAddress(user.addressId);
          return merge(user, address);
        }
      `}
    />

    <Code
      lang="clojure"
      source={`
        (defn getUserAndAddress [id]
          (let [user (getUser id)
                address (getAddress (user :addressId))]
            (merge user address)))
      `} 
    />


    <Code
      lang="clojure"
      title="Async Let"
      source={`
        (defn getUserAndAddress [id]
          (async-let [user (getUser id)
                      address (getAddress (user :addressId))]
            (merge user address)))
      `}
    />

    <Code
      lang="clojure"
      title="Async Await"
      source={`
        (defmacro async-let
          [bindings & body]
          (->> (reverse (partition 2 bindings))
               (reduce (fn [acc [l r]]
                         \`(bind (promise ~r) (fn [~l] ~acc)))               
                       \`(promise (do ~@body)))))
      `}
    />

    <Points size={4} color="green" title='Other "Language" Features'>
      <Point text="Type Checking" />
      <Point text="Pattern Matching" />
      <Point text="Graphql" />
      <Point text="core.async - (Google Go)" />
      <Point text="core.spec" />
    </Points>

    <Headline text="Working in Clojure" />

    <Points title="Out of the Box">
      <Point text="Javascript Interop" />
      <Point text="Minified" />
      <Point text="Source Mapped" />
      <Point text="Dead Code Eliminated" />
      <Point text="Auto Building" />
      <Point text="Browser Repl" />
    </Points>

    <Points color="green" title="Third Party">
      <Point text="Hot Reloading That Actually Works" />
    </Points>

    <Headline color="blue" text="Demo" />

    <Headline text="Moving Beyond" />


  </Presentation>
