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
  ImageSlide,
  Text,
  QuoteSlide,
} from "./library";


// Require CSS
require('normalize.css');

const images = {
  me: require("./images/me.jpg"),
  hashTree: require("./images/hash-tree.svg")
};


preloader(images);
require("./langs")




export default () =>
  <Presentation>
    <Headline
      textAlign="left"
      size={3}
      subtextSize={4}
      text="Meander"
      subtext="Declarative Explorations at the Limits of FP" />





    <Code
      lang="javascript"
      source={`
        let temp = [];
        for (let i = 0; i < arr.length; i ++) {
          temp.push(i * 2)
        }
      `}
     />

     <Code
      lang="javascript"
      source={`
        let temp = [];
        for (let i = 0; i < arr.length; i ++) {
          if (i % 2 === 0) {
            temp.push(i * 2)
          }
        }
      `}
     />

    <Code
      lang="javascript"
      source={`
        let temp = [];
        for (let i = 0; i < arr.length; i ++) {
          if (i % 2 === 0) {
            temp.push(i * 2)
          }
        }
      `}
     />

     <Code
      lang="javascript"
      source={`
        let total = 0;
        for (let i = 0; i < arr.length; i ++) {
          if (i % 2 === 0) {
            total += x*2
          }
        }
      `}
     />


    <Code
      lang="clojure"
      source={`
          (reduce + 0 (filter even? (map (parital * 2) coll)))
      `}
     />


    <Code
      lang="clojure"
      source={`
          (->> coll
               (map (parital * 2))
               (filter even?)
               (reduce + 0))
      `}
     />


    <Points title="What Have We Gained?">
      <Point text="More Readable" />
      <Point text="Declarative Approach" />
      <Point text="About What Not How" />
    </Points>


    <Headline
      color="green"
      textAlign="left"
      text="Is This That Much Better?" />


    <Code
      title="Clean Pipeline"
      lang="clojure"
      source={`
          (->> coll
               (map step1)
               (mapcat step2)
               (partition 2)
               (filter foo?)
               (map (partial extend-bar bar))
               (replace-foo-bar))
      `}
     />



    <Headline
      color="blue"
      textAlign="left"
      text="Still Playing Computer" />


    <TwoColumn
      title="Example Problem"
      color="magenta"
      align="left"
      left={
        <Code
          noSlide
          lang="clojure"
          source={`
           {:name "Jimmy"
            :address
            {:address1 "123 street ave"
             :address2 "apt 2"
             :city "Townville"
             :state "IN"
             :zip "46203"}}}
          `}
        />
      }
      right={
        <Code
          noSlide
          lang="clojure"
          source={`
            {:name "Jimmy"
             :address {:line1 "123 street ave"
                       :line2 "apt 2"}
             :city-info {:city "Townville"
                         :state "IN"
                         :zipcode "46203"}}
          `}
        />
      } 
    />




    <Code
      lang="clojure"
      source={`
        (let [address (:address person)]
          {:name (:name person)
           :address {:line1 (:address1 address)
                     :line2 (:address2 address)}
           :city-info {:city (:city address)
                       :state (:state address)
                       :zipcode (:zip address)}})

      `}
     />


    <TwoColumn
      left={
        <Code
          noSlide
          lang="clojure"
          source={`
           {:name "Jimmy"
            :address
            {:address1 "123 street ave"
             :address2 "apt 2"
             :city "Townville"
             :state "IN"
             :zip "46203"}}}
          `}
        />
      }
      right={
        <Code
          noSlide
          lang="clojure"
          source={`
            {:name "Jimmy"
             :address {:line1 "123 street ave"
                       :line2 "apt 2"}
             :city-info {:city "Townville"
                         :state "IN"
                         :zipcode "46203"}}
          `}
        />
      } 
    />


    <TwoColumn
      left={
        <Code
          noSlide
          lang="clojure"
          source={`
           {:name ?name
            :address
            {:address1 ?address1
             :address2 ?address2
             :city ?city
             :state ?state
             :zip ?zip}}}
          `}
        />
      }
      right={
        <Code
          noSlide
          lang="clojure"
          source={`
            {:name "Jimmy"
             :address {:line1 "123 street ave"
                       :line2 "apt 2"}
             :city-info {:city "Townville"
                         :state "IN"
                         :zipcode "46203"}}
          `}
        />
      } 
    />


    <TwoColumn
      left={
        <Code
          noSlide
          lang="clojure"
          source={`
           {:name ?name
            :address
            {:address1 ?address1
             :address2 ?address2
             :city ?city
             :state ?state
             :zip ?zip}}}
          `}
        />
      }
      right={
        <Code
          noSlide
          lang="clojure"
          source={`
            {:name ?name
             :address {:line1 ?address1
                       :line2 ?address2}
             :city-info {:city ?city
                         :state ?state
                         :zipcode ?zip}}
          `}
        />
      } 
    />

    <Code
      lang="clojure"
      source={`
        (match person
          {:name ?name
           :address 
           {:address1 ?address1
            :address2 ?address2
            :city ?city
            :state ?state
            :zip ?zip}}
          
          {:name ?name
           :address {:line1 ?address1
                     :line2 ?address2}
           :city-info {:city ?city
                       :state ?state
                       :zipcode ?zip}})

      `}
     />






    <BlankSlide />


    <Points title="Serverless">
      <Point text="Modeled as input and output" />
      <Point text="Immutable Deploys" />
      <Point text="Verb Oriented" />
    </Points>

    <Headline
      textAlign="left"
      text="Messy Distributed World" />



    <TwoColumn
      left={
        <>
         <Headline color="cyan" size={4} textAlign="center" noSlide text="About Me" />
         <Image src={images.me} />
       </>
      }
      right={
        <>
          <Text textColor="blue" textSize={60} textAlign="left">Jimmy Miller</Text>
          <Points noSlide styleContainer={{paddingTop: 10}}>
            <Point textSize={40} text="Self Taught" /> 
            <Point textSize={40} text="Senior Developer - Adzerk" /> 
            <Point textSize={40} text="FP Nerd" />
          </Points>
        </>
      } 
    />

    <Code
      lang="clojure"
      title="Source Example"
      source={`
        (source example)
      `}
    />

  </Presentation>
