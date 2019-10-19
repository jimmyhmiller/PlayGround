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
  Dark as Slide,
} from "./library";


// Require CSS
require('normalize.css');

const images = {
  me: require("./images/me.jpg"),
  neuralNet: require("./images/neural-network.png"),
  graph: require("./images/graph.png"),
  linear: require("./images/linear.png"),
  sixthDegree: require("./images/sixth-degree.png"),
  sin: require("./images/sin.png"),
  randomPoints: require("./images/random-points.png"),
  fizzbuzz: require("./images/fizzbuzz.png"),
  approximate: require("./images/approximate.png"),
  wann: require("./images/wann.png"),
  bipedal: require("./images/bipedal.png"),
  guac: require("./images/guac.png")
};


preloader(images);
require("./langs")




export default () =>
  <Presentation>
    <Headline
      textAlign="left"
      size={3}
      text="Live Coding Neural Networks" />

    <TwoColumn
      left={
        <>
         <Headline color="cyan" size={4} textAlign="center" noSlide text="About Me" />
         <Image height={300} src={images.me} />
       </>
      }
      right={
        <>
          <Text textColor="blue" textSize={60} textAlign="left">Jimmy Miller</Text>
          <Points noSlide styleContainer={{paddingTop: 10}}>
            <Point textSize={40} text="Senior Developer - Adzerk" /> 
            <Point textSize={40} text="FP Nerd" />
          </Points>
        </>
      }
    />

    <Points title="Plan of Attack">
      <Point text="Basic Idea" />
      <Point text="Demo" />
      <Point text="Wrap up/Future" />
    </Points>


    <Headline
      color="green"
      textAlign="left"
      text="Neural Networks Aren't Magical" />

    <Headline
      color="magenta"
      textAlign="left"
      text="Neural Networks Aren't Hard" />

    <Headline
      color="yellow"
      textAlign="left"
      text="You don't have to understand everything" />

    <Headline
      color="green"
      textAlign="left"
      text="Neural Networks Approximate Functions" />

    <Headline
      color="magenta"
      textAlign="left"
      text="Everything can be Numbers" />

    <Headline
      color="magenta"
      textAlign="left"
      text="y = f(x)" />


    <ImageSlide
      align="center"
      color="blue"
      src={images.linear}
    />

    <ImageSlide
      align="center"
      color="blue"
      src={images.sixthDegree}
    />

    <ImageSlide
      align="center"
      color="blue"
      src={images.sin}
    />

    <ImageSlide
      align="center"
      color="blue"
      src={images.randomPoints}
    />
    
    <ImageSlide
      align="center"
      color="blue"
      src={images.fizzbuzz}
    />

    <ImageSlide
      align="center"
      color="blue"
      src={images.approximate}
    />


    <ImageSlide
      align="center"
      color="blue"
      src={images.neuralNet}
    />

    <Headline
      color="magenta"
      textAlign="left"
      text="Demo" />

    <Points title="Approaches To Function Approximation">
      <Point text="Feed Forward" />
      <Point text="Convolutional" />
      <Point text="Recurrent" />
    </Points>

    <Points title="Approaches To Training">
      <Point text="Supervised" />
      <Point text="Reinforcement" />
    </Points>

    <Slide>
      <video autoplay controls>
        <source src="https://storage.googleapis.com/quickdraw-models/sketchRNN/wann/mp4/trained_biped.mp4" type="video/mp4" />
      </video>
    </Slide>

    <ImageSlide
      height={670}
      align="center"
      src={images.bipedal}
    />

    <Headline
      color="magenta"
      textAlign="left"
      text="Incredibly Unsatisfying" />

    <ImageSlide
      title="Wann"
      align="center"
      src={images.wann}
    />

    <Headline
      color="blue"
      textAlign="left"
      text="Early Days" />

    <ImageSlide
      align="center"
      src={images.guac}
    />


    <Headline
      color="green"
      textAlign="left"
      text="A Brighter Future" />

    <Headline
      color="magenta"
      textAlign="left"
      text="Language Oriented" />




    <BlankSlide />

  </Presentation>
