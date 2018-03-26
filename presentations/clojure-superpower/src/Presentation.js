
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
      `}
    />

    <Code
      title="Intro To Clojure"
      lang="clojure"
        source={`
        [1 2 3] ; Vector
        (1 2 3) ; List
        {:a 3 :b "adf"} ; Map
        #{1 2 3} ; Set
      `}
    />

    <Code
      title="Code is Data"
      lang="clojure"
      source={`
        (defn add [x y]
          (+ x y))

        (add 3 5)
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

    <Points title="Clojure - Value Proposition" size={4}>
      <Point text="Live Programming" />
      <Point text="Concurrent Programming" />
      <Point text="Scalable Dynamic Programming" />
      <Point text="Expressive Programming" />
    </Points>

    <Headline text="Live Programming" />
    {
    // Initial example
    // Exponential Backoff
    // IOS and Android
    // Server?
    }

    <Headline text="Concurrent Programming" />

    <Points title="Concurrent Programming">
      <Point text="Immutability Makes Concurrency Trivial" />
      <Point text="No Locks" />
      <Point text="Single Threaded Concurrency (even in browser)" />
      <Point text="Actually Practical" />
    </Points>

    {
    // PHash
    // 10000 processes
    }

    <Headline text="Scalable Dynamic Programming" />

    <Points title="Rich Immutable Datastrutures">
      <Point text="List, Vectors, Maps, Sets, etc." />
      <Point text="100+ functions for operating on them" />
      <Point text="Records to define types" />
    </Points>


    <Points title="Clojure Protocols">
      <Point text="Similar to Interfaces" />
      <Point text="Everything in the language is built on it" />
      <Point text="Provides means for first class extension" />
    </Points>

    <Points title="Clojure Spec">
      <Point text="Contract/Validation System" />
      <Point text="Generate Examples" />
      <Point text="Specs for Functions" />
      <Point text="Tests for Free" />
    </Points>

    {
    // Simple Example
    // Generate Examples
    // Generate example fn calls
    // Auto gen tests
    }

    <Headline text="Expressive Programming" />

    <Points title="Code that Writes Code">
      <Point text="Implement Features as Library" />
      <Point text="Express the Problem Your Way" />
      <Point text="No More Waiting Years" />
    </Points>

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

    <Code
      lang="clojure"
      title="Pattern Matching"
      source={`
        (require '[clojure.core.match :refer [match]])

        (doseq [n (range 1 101)]
          (println
            (match [(mod n 3) (mod n 5)]
              [0 0] "FizzBuzz"
              [0 _] "Fizz"
              [_ 0] "Buzz"
              :else n)))
      `}
    />

    <Code
      lang="go"
      title="CSP"
      source={`
        func search() {
            c := make(chan Result)
            go func() { c <- First(query, Web1, Web2) } ()
            go func() { c <- First(query, Image1, Image2) } ()
            go func() { c <- First(query, Video1, Video2) } ()
            timeout := time.After(80 * time.Millisecond)
            for i := 0; i < 3; i++ {
                select {
                case result := <-c:
                    results = append(results, result)
                case <-timeout:
                    return results
                }
            }
            return results
        }
      `}
    />

    <Code
      maxWidth={1200}
      lang="clojure"
      title="CSP"
      source={`
        (use 'core.async)

        (defn search [query]
          (let [c (chan)
                t (timeout 80)]
            (go (>! c (<! (fastest query web1 web2))))
            (go (>! c (<! (fastest query image1 image2))))
            (go (>! c (<! (fastest query video1 video2))))
            (go (loop [i 0 ret []]
                  (if (= i 3)
                    ret
                    (recur (inc i) (conj ret (alt! [c t] ([v] v)))))))))
      `}
    />

    <Points title="Other Features" size={4}>
      <Point text="Static Types" />
      <Point text="Logic Programming" />
      <Point text="Sql as Data Structures" />
      <Point text="Destructuring" />
      <Point text="Elm Style Error Messages" />
      <Point text="Algebraic Data Types" />
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
