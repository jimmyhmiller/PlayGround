
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

import CodeSlide from 'spectacle-code-slide';


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

const handleIndent = (source) =>
  removeIndent(detectIndent(source), source).trim()

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
    <Deck transition={["slide"]} transitionDuration={0} progress="none">
      {children}
    </Deck>
  </Spectacle>

export default () =>
  <Presentation>
    <Headline 
      size={2}
      caps={false}
      subtextSize={3}
      text="Clojure(Script) for Javascript Developers" />

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

    <CodeSlide
      className="datalang"
      lang="clojure"
      code={handleIndent(`
        1
        2
        3

        1.0
        123.1234
        3.14

        "string"
        "jimmy"
        "test"

        false
        true

        nil

        :test
        :a
        :name

        +
        def
        map

        (1 2 3 4 5)
        ("test", "test1", "test2")
        (false 2 2.1 "test")
        (def x 2)

        [1 false "stuff"]
        [1 2 3]
        [false, x, 30]

        {:a 2
         :b 3}

        {:type "Warrior"
         :attack 3 
         :defense 5
         :languages [:common :elven]}

        (defn double [x]
          (* x 2))

        (get {:a 2} :a)
      `)}
      ranges={[
        { loc: [0, 0], title: "A Data Language" },
        { loc: [0, 3], note: "integer"},
        { loc: [4, 7], note: "float"},
        { loc: [8, 11], note: "string"},
        { loc: [12, 14], note: "boolean"},
        { loc: [15, 16], note: "nil (null)"},
        { loc: [17, 20], note: "keyword"},
        { loc: [21, 24], note: "symbol"},
        { loc: [25, 29], note: "list"},
        { loc: [30, 33], note: "vector"},
        { loc: [34, 41], note: "map"},
        { loc: [42, 46], note: "mixed all together"},
      ]}
    />

    <Code
      lang="clojure"
      source={`
        (println "Hello World!")
        (* 2 2)
        (double 3)
        (if true 2 3)
        (def x 2)
      `}
    />

    <Headline
      color="blue"
      text="((Parenthesis) are scary!!!)" />

    <Code
      title="A better Javascript"
      lang="javascript"
      source={`
      // ugly
      map(filter(_.range(100), even), add2) 

      // better
      _.chain(100)
        .range() 
        .filter(even)
        .map(add2)
        .value()

      `} 
    />

    <Code
      title="A better Javascript"
      lang="javascript"
      source={`
      // ugly
      escape(stripFooter(stripHeader(data)))

      // better?
      var dataNoHeader = stripHeader(data);
      var dataNoHeadFoot = stripFooter(dataNoHeader);
      var escapedData = espace(dataNoHeadFoot);
      `} 
    />

    <Code
      title="A better Javascript"
      lang="clojure"
      source={`
      ; ugly
      (map (filter even (range 100)) add2) 

      ; better
      (->> 100
           range
           (map add2)
           (filter even?))

      `} 
    />

    <Code
      title="A better Javascript"
      lang="clojure"
      source={`
      ; ugly
      (escape (stripFooter (stripHeader data)))

      ; better
      (->> data
           stripHeader
           stripFooter
           escape)
      `} 
    />

    <Code
      title="A better Javascript"
      lang="javascript"
      source={`
        var x = ...
        console.log(x.a.b) // bad

        console.log(x && x.a && x.a.b) // good
      `} 
    />

    <Code
      title="A better Javascript"
      lang="clojure"
      source={`
        (def x ...)
        (println (-> x :a :b))
      `} 
    />

    <Code
      title="A better ES6/ES+"
      lang="javascript"
      source={`
        function ([x, y]) {
          return x + y;
        }
        function ({ x, y }) {
          return x + y;
        }
      `} 
    />

    <Code
      title="A better ES6/ES+"
      lang="clojure"
      source={`
        (fn [[x y]]
          (+ x y))
        (fn [{:keys [x y]}]
          (+ x y))
      `} 
    />

    <Code
      title="A better ES6/ES+"
      lang="clojure"
      source={`
        (fn+ [x y] 
          (+ x y))
      `} 
    />

    <Code
      title="A better ES6/ES+"
      lang="clojure"
      source={`
        (defmacro fn+ [bindings & body]
          \`(fn [{:keys ~bindings}] ~@body))
      `} 
    />

    <Code
      title="A better ES6/ES+"
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
      title="A better ES6/ES+"
      lang="javascript"
      source={`
        function getUserAndAddress(id) {
          return getUser(id)
            .then(user => getAddress(user.addressId)
                .then(address => merge(user, address));
        }
      `} 
    />

    <Code
      title="A better ES6/ES+"
      lang="javascript"
      source={`
        async function getUserAndAddress(id) {
          const user = await getUser(id);
          const address = await getAddress(user.addressId);
          return merge(user, address);
        }
      `} 
    />

    <Code
      title="A better ES6/ES+"
      lang="clojure"
      source={`
        (defn getUserAndAddress [id]
          (let [user (getUser id)
                address (getAddress (user :id))]
            (merge user address)))
      `} 
    />

    <Code
      title="A better ES6/ES+"
      lang="clojure"
      source={`
        (defn getUserAndAddress [id]
          (async-let [user (getUser id)
                      address (getAddress (user :id))]
            (merge user address)))
      `} 
    />

    <Code
      title="A better ES6/ES+"
      lang="clojure"
      source={`
        (defmacro async-let
          [bindings & body]
          (->> (reverse (partition 2 bindings))
               (reduce (fn [acc [l r]]
                         \`(bind (promise ~r) (fn [~l] ~acc)))               
                       \`(promise (do ~@body)))))
      `} 
    />


  




  </Presentation>
