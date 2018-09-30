
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
      text="Eliminating Bugs with Types" />

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
      text="Eliminating Bugs with Types" />

    <Headline
      color="blue"
      text="Types Have gotten a bad name" />

    <Points title="Problem With Types">
      <Point text="Verbosity" />
      <Point text="Inheritence" />
      <Point text="Subtyping" />
    </Points>

    <Points title="Uses of Types">
      <Point text="Catching Mistakes" />
      <Point text="Making Programs Run Faster" />
      <Point text="Enforcing Invariants" />
    </Points>

    <Code
      title="Example"
      lang="elm"
      source={`
        type alias Survey = 
        { quesions : List String
        , answers : List (Maybe String)
        }
      `} />

   

    <BlankSlide />

  </Presentation>
