
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
  Text,
} from "@jimmyhmiller/spectacle";

import CodeSlide from 'spectacle-code-slide';

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
  removeFirst(source.split("\n"))
        .map(s => s.substring(indent, s.length))
        .join("\n")

const removeFirst = (arr) => {
  arr.shift();
  return arr;
}

const autoRemove = (source) => {
  return removeIndent(detectIndent(source), source)
}

const BlankSlide = withSlide(() => {
  return <span />;
})

const Code = withSlide(({ source, lang, title, textSize, headlineSize, indent }) => {
  const spaces = indent || detectIndent(source);
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
      size={2} 
      caps={false}
      subtextSize={2}
      textAlign="left"
      text="Clojure"
      subtext="A Computational Lens" />

    <Points title="How Clojure Changed My Perspective" size={4}>
      <Point text="On Programming" />
      <Point text="On OOP" />
      <Point text="On Static Typing" />
      <Point text="On Dynamic Typing" />
    </Points>



    <Headline 
      size={1}
      textAlign="left"
      color="yellow"
      text="On Programming" />

    <Headline 
      color="green"
      size={2}
      caps={false}
      textAlign="left"
      text="Syntax isn't THAT important" />

      <Code
      title="Syntax Wars"
      lang="javascript"
      source={`
        public class Thing 
        {

        }

        public class Thing {

        }
      `}
    />

    <Points title="Syntax Wars" size={4}>
      <Point text="Significant Whitespace" />
      <Point text="Semicolons" />
      <Point text="Trailing Commas" />
      <Point text="Lining up args" />
    </Points>

    <Headline
      color="blue"
      text="Syntax Doesn't need to be beautiful" />

    <Headline
      text="Clojure doesn't have beautiful syntax" />

    <Headline
      text="Clojure's syntax is the best" />

    <Code
      title="Datatypes"
      lang="clojure"
      source={`
        1 ; numbers
        "jimmy" ; string
        :thing ; keyword
        x ; symbol
        true ; boolean
        (1 2 3) ; list
        [1 2 3] ; vector
        {:a 1 :b 3} ; map
      `}
    />

    <Code
      title="Code is Data"
      lang="clojure"
      source={`
        (def x 2)

        (defn add [x y]
          (+ x y))

        (add 12 3) ; 15
      `}
    />

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
      `}
    />


    <Code
      title="Code is Data"
      lang="clojure"
      indent={2}
      source={`
        Verb     Nouns     
         │      ┌─────┐    
         │      │     │    
         ▼      ▼     ▼    
       (def     x     2)   
       ▲               ▲   
       └───────────────┘   
             List          
      `}
    />

    <Headline 
      color="green"
      size={2}
      textAlign="left"
      text="Clojure has no syntax wars" />

    <Headline 
      color="blue"
      size={2}
      textAlign="left"
      text="Languages don't need a lot of features" />

    <Headline
      text="Language features are bad" />

    <Headline
      size={2}
      subtextSize={2}
      textAlign="left"
      text="ES6/ES7/ESNEXT are wonderful"
      subtext="AND A DISASTER" />

    <Headline
      textAlign="left"
      color="green"
      text="Our languages are stagnant" />

    <Headline
      color="blue"
      textAlign="left"
      text="We shouldn't have to wait a decade" />

    <Headline
      textAlign="left"
      text="Clojure enables first class extension" />


    <Code
      lang="javascript"
      title="Async Await"
      source={`
        async function getUserAndAddress(id) {
          const user = await getUser(id);
          const address = await getAddress(user);
          return merge(user, address);
        }
      `}
    />

    <Code
      lang="clojure"
      title="Async Let"
      source={`
        (defn getUserAndAddress [id]
          (async-let [user (getUser id)
                      address (getAddress user)]
            (merge user address)))
      `}
    />


    <Code
      maxWidth={1100}
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


    <Points title="Other Features" size={4}>
      <Point text="Destructuring" /> 
      <Point text="Pattern Matching" /> 
      <Point text="Logic Programming" />
      <Point text="Elm Style Error Messages" />
      <Point text="CSP à la go-lang" />
      <Point text="Static Types" />
      <Point text="Algebraic Data Types" />
    </Points>

    <Headline
      size={1}
      color="yellow"
      textAlign="left"
      text="On OOP" />

    <Headline
      color="blue"
      textAlign="left"
      text="Disclaimer - Nuance is Lost" />

    <Headline
      textAlign="left"
      text="OOP Failed to Deliver" />

    <Points title="OOP promises" size={4}>
      <Point text="Code Reusablity" />
      <Point text="Easier Reasoning" />
      <Point text="Scalable" />
      <Point text="Reduces Coupling" />
    </Points>

    <Points title="OOP Realities" size={4}>
      <Point text="Classes are Concretions" />
      <Point text="Encapsulation Obscures" />
      <Point text="Boilerplate Explodes" />
      <Point text="Code Becomes Fragile" />
    </Points>

    <Points title="Functional Alternative" size={4}>
      <Point text="Pure Functions" />
      <Point text="Immutable Data" />
      <Point text="Composablity" />
    </Points>

    <Points title="Rich Immutable Datastrutures">
      <Point text="List, Vectors, Maps, Sets, etc." />
      <Point text="100+ functions for operating on them" />
      <Point text="Records to define types" />
    </Points>

    <Code
      lang="clojure"
      title="Collections"
      source={`
        {:a :b} ; map
        [:a :b] ; vector
        (:a :b) ; list
        #{:a :b} ; set
      `}
    />

    <Code
      lang="clojure"
      title="Collection Functions"
      source={`
        (->> (range 100)
             (filter even?)
             (map (fn [x] (+ x 2)))
             (reduce +))
        ; 2550
      `}
    />

    <Code
      lang="clojure"
      title="Collection Functions"
      source={`
        (->> (repeatedly #(rand-int 5))
             (take 100)
             (frequencies))

        ; {1 19, 4 21, 2 20, 3 17, 0 23}
      `}
    />

    <Code
      lang="clojure"
      title="Collection Functions"
      source={`
        (def users 
          [{:id 1 :name "foo"} 
           {:id 2 :name "bar"} 
           {:id 3 :name "baz"}])

        (->> users
            (map (juxt :id identity))
            (into {}))

        ;  {1 {:name "foo", :id 1}, 
        ;   2 {:name "bar", :id 2}, 
        ;   3 {:name "baz", :id 3}}
      `}
    />


    <Code
      lang="clojure"
      title="Concurrency for Free"
      source={`
        (time (count (map wait-100ms (range 100))))

        "Elapsed time: 10260.159812 msecs"
        100
      `}
    />

    <Code
      lang="clojure"
      title="Concurrency for Free"
      source={`
        (time (count (pmap wait-100ms (range 100))))

        "Elapsed time: 413.617127 msecs"
        100
      `}
    />

    <Headline
      size={1}
      textAlign="left"
      color="yellow"
      text="On Static Typing" />

    <Headline
      color="blue"
      textAlign="left"
      text="Disclaimer: Static Types are Great" />

    <Headline
      color="green"
      textAlign="left"
      text="Not as needed in a Functional Language" />

    <Headline
      textAlign="left"
      text="Limit Expressiveness" />

    <Headline
      textAlign="left"
      text="Clojure Spec as an Alternative" />


    <Headline
      size={1}
      textAlign="left"
      color="yellow"
      text="On Dynamic Typing" />


    <Headline
      color="blue"
      textAlign="left"
      text="Dynamic Languages Lack Structure" />

    <Headline
      color="green"
      textAlign="left"
      text="Dynamic Languages Lack Abstraction" />


    <Points title="Clojure Protocols">
      <Point text="Similar to Interfaces" />
      <Point text="Everything in the language is built on it" />
      <Point text="Provides means for first class extension" />
    </Points>


    <Code
      lang="clojure"
      title="Protocols"
      source={`
        (defprotocol Speak
          (speak [this]))
      `}
    />

    <Code
      lang="clojure"
      title="Protocols"
      source={`
        (defrecord Bird []
          Speak
          (speak [this] "tweet"))

        (defrecord Dog []
          Speak
          (speak [this] "bark"))

        (speak (Bird.)) ; tweet
        (speak (Dog.)) ; bark
      `}
    />

    <Code
      lang="clojure"
      title="Protocols"
      source={`
        (defprotocol ToJson
          (toJson [this]))
      `}
    />

    <Code
      lang="clojure"
      source={`
        (extend-protocol ToJson
          java.lang.Number
          (toJson [this] (JsonPrimitive. this))
          
          java.lang.String
          (toJson [this] (JsonPrimitive. this))
          
          java.lang.Boolean
          (toJson [this] (JsonPrimitive. this))
          
          clojure.lang.IPersistentVector
          (toJson [this] 
              (reduce (fn [arr x]
                        (doto arr (.add (toJson x))))
                      (JsonArray.) this))

      `}
    />

    <Code
      lang="clojure"
      source={`       
        (str (toJson [1 2 "1234" true]))
        ; "[1,2,\\"1234\\",true]"
      `}
    />


    <Code
      lang="clojure"
      title="Multi Methods"
      source={`
        (defmulti area :shape)

        (defmethod area :circle [{:keys [:radius]}]
          (* Math/PI (Math/pow radius 2)))

        (defmethod area :square [{:keys [:side]}]
          (* side side))

        (area {:shape :circle
               :radius 2}) ; 12.566370614359172

        (area {:shape :square
               :side 5}) ; 25
      `}
    />

  <Headline
      color="blue"
      textAlign="left"
      text="All this is great but..." />
 
    <Headline
      color="green"
      textAlign="left"
      text="Live Coding" />
  
    <Points title="What I couldn't show" size={4}>
      <Point text="Live Coding the Web" />
      <Point text="Live Coding IOS and Android" />
      <Point text="Live Coding Remote" />
      <Point text="More Concurrency" />
      <Point text="Generative Testing" />
      <Point text="Time Traveling Database" />
      <Point text="A Whole Lot More" />
    </Points>

    <Points title="Where Does Clojure Shine?" size={4}>
      <Point text="Systems That Deal With Data" />
      <Point text="Complex Business Logic" />
      <Point text="User Interfaces" />
      <Point text="Third Party Library Interop" />
      <Point text="Rapid Development" />
    </Points>

    <Points title="Companies Using Clojure in Prod" size={4}>
      <Point text="Facebook" />
      <Point text="Netflix" />
      <Point text="Apple" />
      <Point text="Amazon" />
      <Point text="Walmart" />
    </Points>

    <Points title="Learning Clojure" size={4}>
      <Point text="Clojure for the Brave and True" />
      <Point text="Joy of Clojure" />
      <Point text="SICP" />
    </Points>






    <BlankSlide />

  </Presentation>
