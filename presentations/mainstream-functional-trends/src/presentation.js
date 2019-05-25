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
  QuoteSlide,
} from "./library";


// Require CSS
require('normalize.css');

const images = {
  me: require("./images/me.jpg"),
  timeClocks: require("./images/time-clocks.png"),
  timeFig1: require("./images/time-fig-1.png"),
  // https://www.cs.rutgers.edu/~pxk/417/notes/clocks/index.html
  logClocks1: require("./images/logical-clocks-1.png"),
  logClocks2: require("./images/logical-clocks-2.png"),
  cap: require("./images/cap.png"),
  calm: require("./images/calm.png"),
};


preloader(images);
require("./langs")

export default () =>
  <Presentation>
    <Headline
      textAlign="left"
      size={3}
      subtextSize={4}
      text="Functional Trends in the Mainstream"
      subtext="From Frontend to Blockchain to Cloud" />

    <TwoColumn
      title="About Me"
      left={<Image src={images.me} />}
      right={
        <div style={{paddingTop: 80}}>
          <Text textColor="blue" textSize={60} textAlign="left">Jimmy Miller</Text>
          <Points noSlide styleContainer={{paddingTop: 10}}>
            <Point textSize={40} text="Self Taught" /> 
            <Point textSize={40} text="Senior Developer - Adzerk" /> 
            <Point textSize={40} text="FP Nerd" />
          </Points>
        </div>
      } 
    />

    <Headline
      textAlign="left"
      size={3}
      subtextSize={4}
      text="Functional Trends in the Mainstream"
      subtext="From Frontend to Blockchain to Cloud and More" />

    <Points title="Highlights">
      <Point text="React, Redux, React Hooks" />
      <Point text="Blockchain and Smart Contracts" />
      <Point text="Distributed Logs (Kafka and Kinesis)" />
      <Point text="New Trends in Databases" />
      <Point text="Serverless" />
    </Points>

    <Points title="React">
      <Point text="Functional sheep in wolves clothing" />
      <Point text="You never make instances of your classes" />
      <Point text="Everything is immutable" />
      <Point text="Initially written in StandardML" />
    </Points>

    <Headline
      textAlign="left"
      color="blue"
      text="UI = f(state)" />

    <Points title="Redux">
      <Point text="Reducers are pure" />
      <Point text="Emulate algebraic data types" />
      <Point text="Single atom state" />
    </Points>

    <Points title="React Hooks">
      <Point text="Seemingly mutable" />
      <Point text="Borrows from alebraic effects" />
      <Point text="True control over effects" />
      <Point text="No monads" />
    </Points>

    <Points title="BlockChain/Smart Contracts">
      <Point text="Hype is gone" />
      <Point text="Serious security issues" />
      <Point text="Immutable data structures" />
      <Point text="Need for non-turing complete language" />
    </Points>

    <Points title="Distributed Logs">
      <Point text="State becomes a pure function of events" />
      <Point text="Full history is maintained" />
      <Point text="Bugs involving historical data are fixable" />
      <Point text="Ridding ourselves of place oriented programming" />
    </Points>

    <Points title="Databases">
      <Point text="Underlying implementations" />
      <Point text="User Programming Model (Faunadb)" />
      <Point text="Increasingly Immutable" />
    </Points>

    <Points title="Serverless">
      <Point text="Modeled as input and output" />
      <Point text="Immutable Deploys" />
      <Point text="Verb Oriented" />
    </Points>

    <Code
      lang="clojure"
      title="Source Example"
      source={`
        (source example)
      `}
    />


    <BlankSlide />

  </Presentation>
