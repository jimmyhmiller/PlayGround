
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
      subtextSize={2}
      subtextCaps={true}
      text="Don't write tests"
      subtext="Generate them" />

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
      text="Don't write tests"
      subtext="Generate them" />

    <Points title="Outline">
      <Point text="Unit Tests" />
      <Point text="Elixir's Property-Based Testing" />
      <Point text="Examples and Potential Future" />
      <Point text="Details and Other Communities" />
    </Points>


    <Points title="What is a Unit Test">
      <Point text="Tests a (typically) small chunk of code" />
      <Point text="Programmer creates examples" />
      <Point text="Asserts output is as expected" />
    </Points>

    <Code
      title="Unit Test Example"
      lang="elixir"
      source={`
        test "Test Reverse" do
          assert reverse([]) == []
          assert reverse([1]) == [1]
          assert reverse([1,2,3]) == [3,2,1]
          assert reverse(0..10) == Enum.to_list 10..0
        end
      `} />

    <Points title="Unit Test Problems">
      <Point text="Lots of code" />
      <Point text="Don't test whole space" />
      <Point text="Tend towards testing implementation details" />
    </Points>

    <Code
      title="Property-Based Test Example"
      lang="elixir"
      source={`
        property "Reverse a reverse doesn't change" do
          check all list <- list_of(integer()) do
            assert reverse(reverse(list)) == list
          end
        end
      `} />

    <Code
      title="Property-Based Test Example"
      lang="elixir"
      source={`
        property "Reverse append is append reverse" do
          check all list1 <- list_of(integer()),
                    list2 <- list_of(integer()) do
            assert reverse(list1 ++ list2) == 
                   reverse(list2) ++ reverse(list1)
          end
        end
      `} />


    <BlankSlide />

  </Presentation>
