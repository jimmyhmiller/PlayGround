
import React from "react";
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
  Text,
  formatCode,
} from "./library";

import CodeSlide from 'spectacle-code-slide';

const images = {
  me: require("./images/me.jpg"),
};

preloader(images);

export default () =>
  <Presentation>
    <Headline
      textAlign="left"
      subtextSize={4}
      subtextCaps={true}
      text="Functional Architecture"
      subtext="Some Promising Approaches" />


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
      textAlign="left"
      subtextSize={4}
      subtextCaps={true}
      text="Functional Architecture"
      subtext="Some Promising Approaches" />

    <Points title="Caveats">
      <Point text="Speculative Talk" />
      <Point text="Architecture Doesn't live in a Vacuum" />
      <Point text="One Size Doesn't Fit All" />
    </Points>

    <Headline
      color="blue"
      text="What is Architecture?" />

    <Code
      title="Example"
      lang="clojure"
      source={`
      `} />

   

    <BlankSlide />

  </Presentation>
