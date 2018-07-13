
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
      subtextSize={2}
      subtextCaps={true}
      text="Unification"
      subtext="Patterns and Queries" />

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
      subtextSize={2}
      subtextCaps={true}
      text="Unification"
      subtext="Patterns and Queries" />

    <Headline
      textAlign="left"
      color="red"
      text="This is a more advanced talk" />

    <Headline
      textAlign="left"
      color="green"
      text="A topic I'm still learning" />

    <Points title="Outline">
      <Point text="What is unification?" />
      <Point text="Building a unifier" />
      <Point text="Building a pattern matcher" />
      <Point text="Building a query engine" />
      <Point text="Other uses of unification" />
    </Points>


    <Headline
      textAlign="left"
      color="green"
      size={2}
      text="Equality in programming is broken" />

    <Code
      title="Equality"
      lang="javascript"
      source={`
        x = 2
        y = x
        z = y

        // What does z equal?
      `} />

    <Code
      title="Equality"
      lang="javascript"
      source={`
        2 = x
        x = y
        y = z 

        // What does z equal?
      `} />

    <Code
      title="Equality"
      lang="javascript"
      source={`
        x = y
        y = z
        z = 2

        // What does x equal?
      `} />

    <Headline
      textAlign="left"
      color="yellow"
      size={2}
      text="Unification takes equality seriously" />


    <Code
      title="Solving Sudoku"
      lang="clojure"
      source={`
        (defn sudokufd [hints]
          (let [vars (repeatedly 81 lvar) 
                rows (->> vars (partition 9) (map vec) (into []))
                cols (apply map vector rows)
                sqs  (for [x (range 0 9 3)
                           y (range 0 9 3)]
                       (get-square rows x y))]
            (run 1 [q]
              (== q vars)
              (everyg #(infd % (domain 1 2 3 4 5 6 7 8 9)) vars)
              (init vars hints)
              (everyg distinctfd rows)
              (everyg distinctfd cols)
              (everyg distinctfd sqs))))
      `} />









    <BlankSlide />

  </Presentation>
