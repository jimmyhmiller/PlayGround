
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
        x = y
        y = z
        z = 2

        // What does x equal?
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

    <Headline
      textAlign="left"
      color="yellow"
      size={2}
      text="Unification takes equality seriously" />

    <Headline
      textAlign="left"
      color="green"
      size={2}
      text="Pattern Matching" />

    <Code
      title="Simple Patterns"
      lang="clojure"
      source={`
        (match true
               true false
               false true)
        ; false

        (match :name
               :otherName :otherThing
               :name :thing)
        ; :thing
      `} />

    <Code
      title="Patterns"
      lang="clojure"
      source={`
        (match [1 2 3]
               [x]     {:x x}
               [x y]   {:x x :y y}
               [x y z] {:x x :y y :z z})
        
        ; {:x 1 :y 2 :z 3}
      `} />

    <Code
      title="Patterns"
      lang="clojure"
      source={`
        (match [1 2 1]
               [x y x] {:x x :y y}
               [x y z] {:x x :y y :z z})
        
        ; {:x 1 :y 2}
      `} />


    <Code
      title="Patterns"
      lang="clojure"
      source={`
        (defmatch fib
          [0] 0
          [1] 1
          [n] (+ (fib (- n 1))
                 (fib (- n 2))))
      `} />

    <Code
      title="Patterns"
      lang="clojure"
      source={`
        (defmatch get-x
          [x] x
          [x y] x
          [x y z] x)

        (get-x 1)
        (get-x 1 2)
        (get-x 1 2 3)
      `} />

    <Headline
      textAlign="left"
      color="green"
      size={2}
      text="Building" />


    <Code
      title="Queries"
      lang="clojure"
      source={`
        (def db
          [[1 :age 26]
           [1 :name "jimmy"]
           [2 :age 26]
           [2 :name "steve"]
           [3 :age 24]
           [3 :name "bob"]
           [4 :address 1]
           [4 :address-line-1 "123 street st"]
           [4 :city "Indianapolis"]])
      `} />


    <Code
      title="Queries"
      lang="clojure"
      source={`
        (q {:find {:name name}
            :where [[_ :name name]]}
            db)

        ; ({:name "jimmy"} 
        ;  {:name "steve"} 
        ;  {:name "bob"})
      `} />


    <Code
      title="Queries"
      lang="clojure"
      source={`
        (q {:find {:name name
                   :age age}
            :where [[e :name name]
                    [e :age age]]}
           db)

        ; ({:name "jimmy", :age 26}
        ;  {:name "steve", :age 26}
        ;  {:name "bob", :age 24})
      `} />


    <Code
      title="Queries"
      lang="clojure"
      source={`
        (q {:find {:name1 name1
                   :name2 name2}
            :where [[e1 :name name1]
                    [e2 :name name2]
                    [e1 :age age]
                    [e2 :age age]]}
           db)

        ; ({:name1 "jimmy", :name2 "jimmy"}
        ;  {:name1 "jimmy", :name2 "steve"}
        ;  {:name1 "steve", :name2 "jimmy"}
        ;  {:name1 "steve", :name2 "steve"}
        ;  {:name1 "bob", :name2 "bob"})
      `} />

    <Code
      title="Queries"
      lang="clojure"
      source={`
        (q {:find {:name name
                   :address-line-1 address-line-1
                   :city city}
            :where [[e :name name]
                    [a :address e]
                    [a :address-line-1 address-line-1]
                    [a :city city]]}
           db)

        ; ({:name "jimmy",
        ;   :address-line-1 "123 street st",
        ;   :city "Indianapolis"})
      `} />

    <Headline
      textAlign="left"
      color="green"
      size={2}
      text="Building" />

    <Headline
      textAlign="left"
      size={2}
      text="Unifications Other Uses" />

    <Headline
      textAlign="left"
      size={2}
      color="blue"
      text="Logic Programming" />

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



    <Points title="Programming Languages">
      <Point text="Non-lexical lifetimes" />
      <Point text="Trait System" />
      <Point text="Type Inference" />
      <Point text="Dependent types" />
    </Points>


    <Code
      title="Type Inference"
      lang="clojure"
      source={`
        (defn typedo [c x t]
          (conda
            ((lvaro x) (findo x c t))
            ((matche [c x t]
               ([_ [[?x] :>> ?a] [?s :> ?t]]
                  (exist [l]
                    (conso [?x :- ?s] c l)
                    (typedo l ?a ?t)))
               ([_ [:apply ?a ?b] _]
                  (exist [s o]
                    (typedo c ?a [s :> t])
                    (typedo c ?b s)))))))
      `} />

    <Points title="Business Logic">
      <Point text="Decompose Rules" />
      <Point text="Provide Explanations" />
      <Point text="Ignore Ordering" />
      <Point text="No Control Flow" />
      <Point text="Foundation for tooling" />
    </Points>

    <Code
      title="Rules"
      lang="clojure"
      source={`
        (defrule player-in-range
          [?p1 <- Player (= reach ?reach)]
          [?p2 <- Player (<= (distance ?p1 ?p2) ?reach)]
          
          => (insert! (->InRange ?p1 ?p2)))
      `} />

    <Code
      title="Rules"
      lang="clojure"
      source={`
        (defrule provoke-attack-of-opportunity
          [InRange (= x ?target) (= y ?grappler)]
          [Grapple
           (= target ?target)
           (= grappler ?grappler)
           (= false (:improved-grapple ?grappler))]
          
          => (insert! (->AttackOpportunity ?target ?grappler)))
      `} />

    <Code
      title="Rules"
      lang="clojure"
      maxWidth={1300}
      source={`
        rule rules.core/provoke-attack-of-opportunity
          executed
            (do (insert! (->AttackOpportunity ?target ?grappler)))
          with bindings
             {:?target #Player{:name Baron}
              :?grappler #Player{:name Lorc}
          because
             #InRange{:x #Player{:name Baron}
                      :y #Player{:name Lorc}}
               is a rules.entities.InRange
               where [(= x ?target) (= y ?grappler)]
          ...
      `} />

    <Points title="Not Theoretical">
      <Point text="healthfinch uses a rules engine" />
      <Point text="200+ rules around medications" />
      <Point text="Actively working on better tooling" />
      <Point text="Provides explanations of why and why not" />
    </Points>

    <Headline
      textAlign="left"
      color="orange"
      size={2}
      text="FP is full of amazing things" />















    <BlankSlide />

  </Presentation>
