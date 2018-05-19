
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
} from "./library";

const images = {
  me: require("./images/me.jpg"),
};

preloader(images);

export default () =>
  <Presentation>
    <Headline
      textAlign="left" 
      text="Practical Functional Refactoring" />

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
      text="Practical Functional Refactoring" />
    
    <Points title="Limitations">
      <Point text="Code examples can't be large" />
      <Point text="Can't cover all cases" />
      <Point text="Can't address fighting frameworks" />
    </Points>

    <Headline
      color="yellow"
      textAlign="left"
      text="Widely Applicable Refactoring" />

    <Headline
      textAlign="left"
      text="Eliminate Loops" />

    <Code
      title="Eliminate Loops"
      lang="javascript"
      source={`
        var oldArray = [1,2,3]
        var newArray = [];
        for (var i = 0; i < oldArray.length; i++) {
          if (oldArray[i] % 2 === 0) {
            newArray.push(oldArray[i] * 2);
          }
        }
      `} />

    <Code
      title="Eliminate Loops"
      lang="javascript"
      source={`
        var oldArray = [1,2,3]
        var newArray = oldArray
          .filter(x => x % 2 === 0)
          .map(x => x * 2)
      `} />

    <Code
      title="Eliminate Loops"
      lang="javascript"
      source={`
        function sumOfActiveScores(team) {
          var totalScore = 0;
          for (player of team) {
            if (player.active) {
              for (score of player.scores) {
                if (score !== null) {
                  totalScore += score;
                }
              }
            }
          }
          return totalScore;
        }
      `} />

    <Code
      title="Eliminate Loops"
      lang="javascript"
      source={`
        function sumOfActiveScores(team) {
          return team
            .filter(player => player.active)
            .flatMap(player => player.scores)
            .filter(score => score !== null)
            .reduce((total, score) => total + score, 0);
        }
      `} />




    <BlankSlide />

  </Presentation>
