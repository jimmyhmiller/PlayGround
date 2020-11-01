// Import React
import React from 'react';


import { 
  Presentation,
  Code,
  Headline,
  TwoColumn,
  Point,
  Points,
  preloader,
  BlankSlide,
  Image,
  ImageSlide,
  Text,
  QuoteSlide
} from "./library";


// Require CSS
require('normalize.css');

const images = {
  me: require("./images/me.jpg"),
  falcon: require("./images/falcon.jpg"),
};


preloader(images);
require("./langs")




export default () =>
  <Presentation>
    <Headline
      textAlign="left"
      size={3}
      subtextSize={4}
      text="Paradigms Without Progress"
      subtext="Kuhnian Reflections on Programming Practice" />


    <Headline
      color="green"
      textAlign="left"
      text="Interminable Debates" />


    <Headline
      color="blue"
      textAlign="left"
      text="Meta" />

    <Headline
      textAlign="left"
      size={3}
      subtextSize={4}
      text="The Paradigms of Programming"
      subtext="Robert  W. Floyd (1978)" />

    {/*TODO: Change colors*/}
    <Headline
      textAlign="left"
      size={3}
      subtextSize={4}
      text="The Structure of Scientific Revolutions"
      subtext="Thomas S. Kuhn (1962)" />

    <Points title="Outline">
      <Point text="Kuhnian Paradigms" />
      <Point text="Peter Naur" />
      <Point text="Other Contributions" />
      <Point text="The Future" />
    </Points>

    <Headline
      color="cyan"
      textAlign="left"
      text="The March of Progress" />


    <Points title="Paradigm">
      <Point text="Ontology" />
      <Point text="Theory" />
      <Point text="Methods" />
      <Point text="Standards" />
      <Point text="Values" />
      <Point text="Relevant Problems" />
      <Point text="Acceptable Solutions" />
    </Points>

    <Headline
      color="cyan"
      textAlign="left"
      text="How Science Really Progresses" />

    <Headline
      color="cyan"
      textAlign="left"
      text="Normal Science" />

    <Headline
      color="cyan"
      textAlign="left"
      text="Crisis" />

    <Headline
      color="cyan"
      textAlign="left"
      text="Revolution" />


    <Headline
      color="cyan"
      textAlign="left"
      text="Breaking Down A Paradigm" />

    <Headline
      color="cyan"
      textAlign="left"
      text="Examplar" />

    <Points title="Exemplar Science">
      <Point text="Aristotle's Physics" />
      <Point text="Newton's Principia Mathematica" />
      <Point text="Darwin's Origin of Species" />
      <Point text="Maxwells's Equations" />
      <Point text="Einstein's Special Relativity" />
    </Points>

    <Points title="Exemplar Programming">
      <Point text="OOP - Smalltalk/Java" />
      <Point text="FP - Lisp/Haskell" />
      <Point text="Imperative - Algol/C" />
    </Points>

    <QuoteSlide
      color="blue"
      text=" It became the exemplar of the new computing, in part, because we were actually trying for a qualitative shift in belief structures — a new Kuhnian paradigm..." />

    <Headline
      color="cyan"
      textAlign="left"
      text="Incommensurability" />

    <Headline
      color="cyan"
      textAlign="left"
      text="Having No Common Measure" />

    <QuoteSlide 
      color="blue"
      text="No neutral algorithm for theory choice, no systematic decision procedure which... must lead each individual in the group to the same decision." />

    <Headline
      color="cyan"
      textAlign="left"
      text="Difficulties Communicating" />

    <Points title="Weird Entities of the Past">
      <Point text="Phlogiston" />
      <Point text="Corpuscle" />
      <Point text="Caloric" />
    </Points>

    <Headline
      color="cyan"
      textAlign="left"
      text="Progress" />


    <Headline
      color="cyan"
      textAlign="left"
      text="Are Programming Paradigms Kuhnian?" />

    <Headline
      color="cyan"
      textAlign="left"
      text="Pre-paradigm" />

    <Headline
      textAlign="left"
      size={3}
      subtextSize={4}
      text="Mathematized Computer Science"
      subtext="Tomas Petricek (2016)" />

    <Headline
      color="cyan"
      textAlign="left"
      text="What Have We Learned?" />


    <Headline
      textAlign="left"
      size={3}
      subtextSize={4}
      text="Programming as Theory Building"
      subtext="Peter Naur (1985)" />

    <QuoteSlide 
      color="blue"
      text="We will misunderstand the difficulties that arise and our attempts to overcome them will give rise to conflicts and frustrations." />

    <Headline
      textAlign="left"
      size={3}
      subtextSize={4}
      text="The Concept of Mind"
      subtext="Gilbert Ryle (1949)" />

    <QuoteSlide
      color="blue"
      text="A main claim of the Theory Building View of programming is that an essential part of any program, the theory of it, is something that could not conceivably be expressed, but is inextricably bound to human beings" />

    <Headline
      color="cyan"
      textAlign="left"
      text="Knowing How vs Knowing That" />

    <Headline
      color="cyan"
      textAlign="left"
      text="Rewrites as Paradigm Change" />

    <Points title="Philosophical Influences">
      <Point text="Turing/Church: Hilberts MetaMathematics" />
      <Point text="Russell: Theory of Types" />
      <Point text="Carnap: The Logical Syntax of Language" />
    </Points>

    <Points title="Philosophical Influences">
      <Point text="SmallTalk: Leibniz's Monads" />
      <Point text="Quine: Referential Transparency" />
      <Point text="Joseph Halpern: Reasoning About Knowledge" />
      <Point text="John McCarthy: Elephant 2000" />
    </Points>

    <Points title="Conceptual Engineering">
      <Point text="Sally Haslanger" />
      <Point text="Herman Cappelen" />
      <Point text="David Chalmers" />
    </Points>

    <Headline
      textAlign="left"
      size={3}
      subtextSize={4}
      text="Patterns of Intention"
      subtext="Michael Baxandall (1987)" />

    <Points title="Abstraction">
      <Point text="Implementation as Semantic Interpretation - William J. Rapaport" />
      <Point text="Making Things Up - Karen Bennett" />
    </Points>


    <Headline
      color="cyan"
      textAlign="left"
      text="Reflecting" />




    <BlankSlide />

  </Presentation>
