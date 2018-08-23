
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


// What are unit tests
// Example
// Problems
// What are property based tests
// Example in lots of languages
// Demo
// Breaking down the parts (Haskell and Clojure)
// Patterns
// Taking it further
// How
// Why

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
      <Point text="Property-Based Testing" />
      <Point text="Patterns for Properties" />
      <Point text="Generating Data" />
      <Point text="Taking this idea further" />
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
      <Point text="Doesn't test the whole space" />
      <Point text="Tend towards testing implementation details" />
    </Points>

    <Points title="What is a Property-Based Test">
      <Point text="Program creates examples" />
      <Point text="Programmer provides properties" />
      <Point text="Program finds minimal failure case" />
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
        check(property(gen.array(gen.int), (xs) => {
           return xs === reverse(reverse(xs))
        }))
      `} />

    <Code
      title="Property-Based Test Example"
      lang="clojure"
      source={`
        (prop/for-all [xs (gen/list gen/int)]
          (= (reverse (reverse xs)) xs))
      `} />

    <Code
      title="Property-Based Test Example"
      lang="python"
      source={`
        @given(st.lists(st.integers()))
        def test_reversing_twice_gives_same_list(xs):
            assert xs == reverse(reverse(xs))
      `} />

    <Headline
      color="yellow"
      text="Demo" />

    <Points title="Property-Based testing parts">
      <Point text="A way to specify properties" />
      <Point text="A way to generate random values" />
    </Points>

    <Headline
      color="blue"
      textAlign="left"
      text="Properties are hard to think of" />

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
      color="green"
      text="Generating Data" />


    <Points title="Static Generation">
      <Point text="Generators specified by types" />
      <Point text="Generation is implicit" />
      <Point text="Shrinking is dependent on types" />
    </Points>

    <Points title="Dynamic Generation">
      <Point text="Build your own generators" />
      <Point text="Generation is explicit" />
      <Point text="Shrinking is based on your generators" />
    </Points>


    <Code
      title="Static Generation"
      lang="haskell"
      source={`
        evenOrOdd :: Int -> Bool
        evenOrOdd x = even x || odd x

        main :: IO ()
        main = quickCheck evenOrOdd

        -- +++ OK, passed 100 tests.
      `} />

    <Code
      title="Static Generation"
      lang="haskell"
      source={`
        data Point = Point Int Int

        -- incorrect (ex: Point 0 0)
        posPoint :: Point -> Point
        posPoint (Point x y) = Point (abs x) (abs y)

        pointIsPos :: Point -> Bool
        pointIsPos (Point x y) = isPos x && isPos y 

        main :: IO ()
        main = quickCheck (pointIsPos . posPoint)
      `} />

    <Code
      title="Static Generation"
      lang="haskell"
      source={`
        • No instance for (Arbitrary Point)
            arising from a use of ‘quickCheck’
        • In a stmt of a 'do' block: quickCheck (pointIsPos . posPoint)
          In the expression: do quickCheck (pointIsPos . posPoint)
          In an equation for ‘main’:
              main = do quickCheck (pointIsPos . posPoint)
           |
        23 |     quickCheck (pointIsPos . posPoint)
           |     ^
      `} />

    <Code
      title="Static Generation"
      lang="haskell"
      source={`
        instance Arbitrary Point where
            arbitrary = do
                x <- arbitrary
                y <- arbitrary
                return (Point x y)
      `} />


    <Code
      title="Static Generation"
      lang="rust"
      source={`
        struct Point {
            x: i32,
            y: i32
        }

        impl Arbitrary for Point {
            fn arbitrary<G: Gen>(g: &mut G) -> Point {
                let x : i32 = i32::arbitrary(g);
                let y : i32 = i32::arbitrary(g);
                Point { x, y } 
            }
        }

      `} />


    <Code
      title="Static Generation"
      lang="rust"
      source={`
          fn pos_point(point : Point) -> Point {
              Point { 
                  x: point.x.abs(),
                  y: point.y.abs()
              }
          }

          quickcheck! {
              fn is_pos(point : Point) -> bool {
                  point.x > 0 && point.y > 0
              }
          }

      `} />

    <Code
      title="Dynamic Generation"
      lang="elixir"
      source={`
        check all list1 <- list_of(integer()),
                  list2 <- list_of(integer()) do
          assert reverse(list1 ++ list2) ==
                   reverse(list2) ++ reverse(list1)
        end
      `} />

    <Code
      title="Dynamic Generation"
      lang="clojure"
      source={`
        (prop/for-all [list1 (gen/list gen/int)
                       list2 (gen/list gen/int)]
           (= (reverse (concat list1 list2))
              (concat (reverse list2)
                      (reverse list1))))
      `} />

      

    <Code
      textSize={24} 
      title="A Little of Both"
      lang="javascript"
      source={`
        @Theory 
        public void testJson(@ForAll @From(JsonValueGenerator.class) JsonValue s1) {
            ...
        }
      `} />

    <Code
      textSize={24}
      title="A Little of Both"
      lang="javascript"
      source={`
        public class JsonValueGenerator extends Generator<JsonValue> {

            public JsonValue generate(SourceOfRandomness random, GenerationStatus status) {
                Integer pickType = random.nextInt(0, 5);
                if (pickType == 0) {
                    return new JsonNumberGenerator().generate(random, status);
                } else if (pickType == 1) {
                    return new JsonStringGenerator().generate(random, status);
                } else if (pickType == 2) {
                    return new JsonBooleanGenerator().generate(random, status);
                } else if (pickType == 3) {
                    return new JsonObjectGenerator().generate(random, status);
                } else if (pickType == 4) {
                    return new JsonArrayGenerator().generate(random, status);
                }
                return JsonValue.NULL;
            }
        }
      `} />

    <Code
      textSize={22}
      title="A Little of Both"
      lang="javascript"
      source={`
        public class DocTest
        {
            [TestInitialize]
            public void Initialize()
            {
                Arb.Register<MyArbitraries>();          
            }
            [TestMethod]
            public void QuickTest()
            {
                Prop.ForAll<Doc>(doc => doc.ToString() != "")
                    .QuickCheckThrowOnFailure();
            }
            private class MyArbitraries
            {
                public static Arbitrary<Doc> Doc()
                {
                    return Gen.Sized(DocGenenerator.Generator).ToArbitrary();
                }    
            } 
        }     
      `} />

    <Headline
      textAlign="left"
      color="yellow"
      text="How Generation Works" />

    <Headline
      textAlign="left"
      color="yellow"
      text="Demo" />

    <Code
      textSize={24}
      title="Api for Generators"
      lang="haskell"
      source={`
        map :: (a -> b) -> Generator a -> Generator b
        bind :: Generator a -> (a -> Generator b) -> Generator b
        constant :: a -> Generator a
      `} />

    <Headline
      textAlign="left"
      color="yellow"
      text="Generators are Monads" />

    <Headline
      textAlign="left"
      color="green"
      text="Taking this idea further" />

    <Headline
      text="Importance" />

    <BlankSlide />

  </Presentation>
