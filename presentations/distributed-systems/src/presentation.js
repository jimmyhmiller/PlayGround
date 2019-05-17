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
  timeClocks: require("./images/time-clocks.png"),
  timeFig1: require("./images/time-fig-1.png"),
  // https://www.cs.rutgers.edu/~pxk/417/notes/clocks/index.html
  logClocks1: require("./images/logical-clocks-1.png"),
  logClocks2: require("./images/logical-clocks-2.png"),
  cap: require("./images/cap.png"),
};


preloader(images);
require("./langs")

export default () =>
  <Presentation>
    <Headline
      textAlign="left"
      size={3}
      subtextSize={4}
      text="Einstein, Hats, and Propaganda"
      subtext="An Introduction to Distrubuted Systems" />

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
      text="Einstein, Hats, and Propaganda"
      subtext="An Introduction to Distrubuted Systems" />

    <Points title="Papers">
      <Point text="Time, Clocks, and the Ordering of Events in a Distributed System" />
      <Point text="Brewer’s conjecture and the feasibility of consistent, available, partition-tolerant web services" />
      <Point text="Keeping CALM: When Distributed Consistency is Easy" />
    </Points>

    <Points title="About the papers">
      <Point text="Incredibly Readable (very little math)" />
      <Point text="Short (8-13 pages)" />
      <Point text="Full of Interesting Insights" />
    </Points>


    <Points title="Approach">
      <Point text="Establish the Problem" />
      <Point text="Draw Some Conclusions" />
      <Point text="Show One Path Forward" />
    </Points>

    <ImageSlide 
      height={700}
      src={images.timeClocks} />


    <Points title="What is a distributed system">
      <Point text="Distinct Processes" />
      <Point text="Spatially Separated" />
      <Point text="Communicate by Exchanging Messages" />
    </Points>

    <QuoteSlide
      text="A network of interconnected computers, such as the ARPA net, is a  distributed system."
    />

    <Headline
      textAlign="left"
      color="blue"
      text="How can we tell the ordering of events?" />

    <Headline
      caps={false}
      textAlign="left"
      text="happens-before relation" />

    <Code
      lang="python"
      title="happens-before definition"
      source={`
        "a" happens before "b" if 
        "a" happened at an earlier time than "b"
      `}
    />

    <Code
      lang="python"
      title="happens-before definition"
      source={`
        if a.time is earlier than b.time:
          a happens_before b
      `}
    />

    <Code
      lang="clojure"
      title="happens-before definition"
      source={`
        (if (is-earlier-than (time a)
                             (time b))
          (happens-before a b))
      `}
    />

    <Code
      lang="java"
      title="happens-before definition"
      source={`
        EventInProcess a = EventInProcess.fromEventFactory("a", Time.fromFactory("yesterday"))
        EventInProcess b = EventInProcess.fromEventFactory("a", Time.fromFactory("today"))
        EarlierThanDeterminerFactoryProxy etdfp = EarlierThanDeterminerFactoryProxy.prxoy();
        etdfp.initialize();
        etdfp.addEventAtTime(a, EventInProcess.class);
        EventIdentifier ea = etdfp.getPreviousId();
        etdfp.addEventAtTime(b, EventInProcess.class);
        EventIdentifier eb = etdfp.getPreviousId();
        if (etdfp.evaluator().isBefore(ea.getTime(), eb.getTime()) {
          etdfp.assertHappensBefore(ea, eb);
        }

      `}
    />

    <Headline
      caps={false}
      textAlign="left"
      color="green"
      text="Time is Relative" />

    <ImageSlide 
      height={700}
      src={images.timeFig1} />

    <Code
      lang="python"
      title="happens-before definition"
      source={`
        def happens_before(a, c):
          if a.process == c.process and a.time < c.time:
            return True

          if a.process != c.process and a.message_sent == c.message_recieved
            return True

          if any(happens_before(a, b) and happens_before(b, c) for b in events):
            return True

          return False

      `}
    />


    <ImageSlide 
      height={700}
      src={images.timeFig1} />


    <Headline
      caps={false}
      textAlign="left"
      color="yellow"
      text="Logical Clocks" />


    <ImageSlide 
      height={400}
      src={images.logClocks1} />

    <ImageSlide 
      height={400}
      src={images.logClocks2} />

    <Headline
      caps={false}
      textAlign="left"
      color="green"
      text="P0 > P1 && P1 > P2" />

    <ImageSlide 
      height={400}
      src={images.logClocks2} />

    <Points title="What we've gained">
      <Point text="Causal Ordering" />
      <Point text="Ability to synchronize without centralizing" />
      <Point text="Basis for writing distributed algorithms" />
    </Points>

    <Points title="What we don't have">
      <Point text="Dynamic Scalability" />
      <Point text="Intuitive ordering" />
      <Point text="Fault Tolerence" />
    </Points>

    <QuoteSlide
      text="We will just observe that the entire concept of failure is only meaningful in the context of physical time."
    />


    <ImageSlide 
      height={700}
      src={images.cap} />


    <Points title="CAP Theorm">
      <Point text="Consistency" />
      <Point text="Availability" />
      <Point text="Partition Tolerence" />
    </Points>

    <QuoteSlide
      text="There must exist a total order on all operations such that each operation looks as if it were complete at a single instant."
    />

    <QuoteSlide
      color="magenta"
      text="Every request recieved by a non-failing node in a system must result in a response."
    />

    <QuoteSlide
      color="blue"
      text="The network will be allowed to lose arbitarily man messages sent from one node to another."
    />


    <BlankSlide />

  </Presentation>
