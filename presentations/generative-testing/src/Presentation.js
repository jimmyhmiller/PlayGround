
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
      <Point text="Doesn't test the whole space" />
      <Point text="Tend towards testing implementation details" />
    </Points>


    <Code
      title="Property-Based Test Example"
      lang="haskell"
      source={`
        reverse_twice_not_changed :: [Int] -> Bool
        reverse_twice_not_changed xs = reverse (reverse xs) == xs
      `} />

    <Code
      title="Property-Based Test Example"
      lang="javascript"
      source={`
        [Property]
        public bool ReverseTwiceNotChanged(int[] xs)
        {
            return xs.Reverse().Reverse().SequenceEqual(xs);
        }
      `} />

    <Code
      title="Property-Based Test Example"
      lang="javascript"
      source={`
        jsc.property("reverse twice not changed", "array int", (arr) => {
          return _.isEqual(reverse(reverse(arr)), arr);
        });
      `} />

    <Code
      title="Property-Based Test Example"
      lang="clojure"
      source={`
        (prop/for-all [xs (gen/list gen/int)]
                      (= (reverse (reverse xs)) xs))
      `} />

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
        no_errors_thrown(my_code())
      `} />

    <Code
      headlineSize={2}
      maxWidth={800}
      title="Conversion code"
      lang="elixir"
      source={`
        is_right_shape(convert(external_data))
      `} />

    <Headline
      color="blue"
      text="Extensions" />

    <Headline
      text="Importance" />









    <BlankSlide />

  </Presentation>