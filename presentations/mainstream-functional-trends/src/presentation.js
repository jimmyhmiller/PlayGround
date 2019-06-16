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
      text="Functional Trends in the Mainstream"
      subtext="From Frontend to Blockchain to Cloud" />

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
      size={3}
      subtextSize={4}
      text="Functional Trends in the Mainstream"
      subtext="From Frontend to Blockchain to Cloud and More" />

    <Points title="Highlights">
      <Point text="React, Redux, React Hooks" />
      <Point text="Blockchain and Smart Contracts" />
      <Point text="Distributed Logs (Kafka and Kinesis)" />
      <Point text="New Trends in Databases" />
      <Point text="Serverless" />
    </Points>

    <Points title="React">
      <Point text="Functional sheep in wolves clothing" />
      <Point text="You never make instances of your classes" />
      <Point text="Everything is immutable" />
      <Point text="Initially written in StandardML" />
    </Points>

    <Headline
      textAlign="left"
      color="blue"
      text="UI = f(state)" />

    <Points title="Clever Adoption Strategy">
      <Point text="Started with Classes" />
      <Point text="Provide Functional State Management" />
      <Point text="Mutation Warnings in Dev" />
      <Point text="Never Mention Functional Programming" />
    </Points>

    <Points title="Larger Effect">
      <Point text="Elm" />
      <Point text="SwiftUI" />
      <Point text="Android Jetpack" />
      <Point text="Flutter" />
      <Point text="Revery" />
    </Points>

    <Points title="Redux">
      <Point text="Reducers are pure" />
      <Point text="Emulate algebraic data types" />
      <Point text="Single atom state" />
    </Points>


    <Points title="Redux">
      <Point text="Business logic is pure" />
      <Point text="Change Can be Modeled Without Mutation" />
      <Point text="Immutablity Provides Direct Benefit" />
    </Points>


    <Points title="React Hooks">
      <Point text="Seemingly mutable" />
      <Point text="Borrows from alebraic effects" />
      <Point text="True control over effects" />
      <Point text="No monads" />
    </Points>


    <Code
      lang="javascript"
      title="React Hooks"
      source={`
        const [count, setCount] = useState(0);
      `}
    />

    <Code
      lang="javascript"
      title="React Hooks"
      source={`
        const [error, setError] = useState(null);
        const [count, setCount] = useState(0);
        useEffect(() => {
          console.log(count);
        }, [count]);
      `}
    />

    <Code
      lang="javascript"
      title="React Hooks"
      source={`
        const [user, setUser] = useState(initialUser);
        useEffect(() => {
          const unsubscribe = subscribeToUpdates(user);
          return unsubscribe;
        }, [user]);
      `}
    />

    <Points title="Lessons to learn">
      <Point text="What, not how" />
      <Point text="Pure functions are great" />
      <Point text="Effects can be lifted to a higher level" />
    </Points>


        <Points title="Difficulties">
      <Point text="Multiple Views of Same Data" />
      <Point text="Async by default" />
      <Point text="New Paradigm" />
    </Points>

    <Points title="Databases">
      <Point text="Underlying implementations" />
      <Point text="User Programming Model (Faunadb)" />
      <Point text="Increasingly Immutable" />
    </Points>

    <Points title="Underlying implementations">
      <Point text="Built on Immutable Logs" />
      <Point text="Lowers lock contention" />
      <Point text="Immutability helps with consensus algorithms" />
      <Point text="Experiments in lattice based KV stores" />
    </Points>

    <Points title="User Programming Model (Faunadb)">
      <Point text="Lisp In your DB" />
      <Point text="Not Turing Complete" />
      <Point text="Distributed, Relational DB" />
      <Point text="Serverless" />
    </Points>


    <Points title="User Programming Model (Faunadb)">
      <Point text="Lisp In your DB" />
      <Point text="Not Turing Complete" />
      <Point text="Distributed, Relational DB" />
      <Point text="Serverless" />
    </Points>


    <Points title="Increasingly Immutable">
      <Point text="Fauna keep versions" />
      <Point text="Datomic and Crux immutable by default" />
      <Point text="RDS allows for 1 second snapshots" />
      <Point text="New Databases are starting to be fully immutable" />
    </Points>


    <Points title="Distributed Logs">
      <Point text="Kafka" />
      <Point text="Kinesis" />
    </Points>

    <Points title="Distributed Logs">
      <Point text="Full history is maintained" />
      <Point text="State becomes a pure function of events" />
      <Point text="Bugs involving historical data are fixable" />
      <Point text="Ridding ourselves of place oriented programming" />
    </Points>

    <Points title="Difference to Queues">
      <Point text="History is Immutable" />
      <Point text="Consume from anywhere" />
      <Point text="Reprocess" />
      <Point text="Multiple consumers allowed by default" />
    </Points>

    <Headline
      textAlign="left"
      color="yellow"
      text="CQRS" />

    <Headline
      textAlign="left"
      color="yellow"
      text="(State, Action) => State" />


    <Points title="Benefits">
      <Point text="Auditability" />
      <Point text="New Features without Planning" />
      <Point text="Inspectable by Default" />
      <Point text="Decoupled systems" />
    </Points>


    <Points title="BlockChain/Smart Contracts">
      <Point text="Hype is gone" />
      <Point text="Serious security issues" />
      <Point text="Immutable data structures" />
      <Point text="Need for non-turing complete language" />
    </Points>

    <ImageSlide
      align="center"
      height={600}
      title="Merkle Tree"
      src={images.hashTree} />

    <Headline
      textAlign="left"
      color="green"
      text="One World Computer" />

    <Points title="Problems">
      <Point text="Infinite loops" />
      <Point text="Error Handling" />
      <Point text="Out of Memory" />
      <Point text="Concurrent by default" />
    </Points>

    <Code
      textSize={24}
      lang="javascript"
      source={`
        contract Balance {
          mapping (address => uint) userBalances;

          function addToBalances() {
            userBalances[msg.sender] += msg.value;
          }

          function withdrawBalance() {
            bool sent = msg.sender.call.value(
              userBalances[msg.value]
            )();

            if (!sent) {
              throw;
            }
            userBalances[msg.sender] = 0;
          }
        }
      `}
    />

    <Code
      textSize={24}
      lang="javascript"
      source={`

        // 100 dollar balance

        contract ExploitWithdrawl {
            function() external { 
              int withdrawn = 0;
              try {
                withdrawn += Balance.withdrawBalance();
              } catch (e) {}
            }
        }
      `}
    />

    <Points title="Functional Benefits">
      <Point text="All Code is Pure" />
      <Point text="Can be Turing Incomplete" />
      <Point text="Encode Guarantees as Types" />
    </Points>


    <Points title="Serverless">
      <Point text="Modeled as input and output" />
      <Point text="Immutable Deploys" />
      <Point text="Verb Oriented" />
    </Points>

    <Headline
      textAlign="left"
      text="Messy Distributed World" />




    <BlankSlide />

    <Code
      lang="clojure"
      title="Source Example"
      source={`
        (source example)
      `}
    />

  </Presentation>
