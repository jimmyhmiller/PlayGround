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
  Text,
  formatCode,
} from "./library";

// Import theme
import createTheme from 'spectacle/lib/themes/default';

// Require CSS
require('normalize.css');

const images = {
  me: require("./images/me.jpg"),
};


preloader(images);
require("./langs")

export default () =>
  <Presentation>
    <Headline
      textAlign="left"
      text="Let your code write itself" />

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
      text="Let your code write itself" />

    <Headline
      textAlign="left"
      color="blue"
      text="Not a gimmick" />

    <Points title="Why?">
      <Point text="Error prone" />
      <Point text="Waste of time" />
      <Point text="What programming is about" />
    </Points>

    <Headline
      textAlign="left"
      color="green"
      text="Programming is thought systemetized" />

    <Headline
      textAlign="left"
      color="magenta"
      text="Languages should let us think, not inhibit us" />

    <Points title="Approach">
      <Point text="Idris - Static" />
      <Point text="Barliman - Dynamic" />
      <Point text="Both experimental" />
    </Points>

    

    <BlankSlide />

  </Presentation>
