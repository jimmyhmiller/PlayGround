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


export default () =>
  <Presentation>
    <Headline
      textAlign="left"
      subtextSize={2}
      subtextCaps={true}
      text="Don't write tests"
      subtext="Generate them" />
    <Headline
      textAlign="left"
      subtextSize={2}
      subtextCaps={true}
      text="Don't write tes"
      subtext="Generate them" />
  </Presentation>
