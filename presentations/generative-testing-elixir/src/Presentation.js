
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
      <Point text="Patterns for Properties" />
      <Point text="Sketching some ideas" />
      <Point text="The importance of the idea" />
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

    <Headline
      color="yellow"
      text="Demo" />

    <Points title="Reaching Understanding">
      <Point text="Breaking Down" />
      <Point text="Removing Sugar" />
      <Point text="Building up" />
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

    <Points title="Disecting Example">
      <Point text="property" />
      <Point text="check" />
      <Point text="all" />
      <Point text="list_of" />
      <Point text="integer" />
    </Points>

    <Points title="Modules">
      <Point text="StreamData" />
      <Point text="ExUnitProperties" />
    </Points>

    <Headline
      color="yellow"
      text="Demo" />

    <Headline
      color="green"
      text="Patterns"
      subtextSize={8}
      subtext="Taken from Andrea Leopardi" />

    <Code
      headlineSize={2}
      maxWidth={800}
      title="Circular Test"
      lang="elixir"
      source={`
        decode(encode(term)) == term
      `} />

    <Code
      headlineSize={2}
      maxWidth={800}
      title="Oracle Code"
      lang="elixir"
      source={`
        my_code() == oracle_code()
      `} />

    <Code
      headlineSize={2}
      maxWidth={800}
      title="Smoke Test"
      lang="elixir"
      source={`
        no_invalid_output(my_code())
      `} />

    <Code
      headlineSize={2}
      maxWidth={800}
      title="Coversion code"
      lang="elixir"
      source={`
        is_right_shape(convert(external_data))
      `} />

    <Headline
      color="blue"
      text="Possible Extensions" />

    <Headline
      text="Importance" />









    <BlankSlide />

  </Presentation>
