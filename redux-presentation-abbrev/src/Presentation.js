
import React, { Component } from "react";
import { createStore } from 'redux';
import { Provider, connect } from 'react-redux';

import {
  CodePane,
  Deck,
  Fill,
  Heading,
  Image,
  Layout,
  ListItem,
  List,
  Slide,
  Text,
  ComponentPlayground,
} from "@jimmyhmiller/spectacle";

import preloader from "@jimmyhmiller/spectacle/lib/utils/preloader";
import { format } from 'prettier-standalone';
import {combineReducers} from 'redux'
import undoable from 'redux-existing-undo';
import { ActionCreators } from 'redux-existing-undo';
import { persistStore, autoRehydrate } from 'redux-persist'
import {storeEnhancer} from 'redux-bug-reporter'
import ReduxBugReporter from 'redux-bug-reporter'
import 'redux-bug-reporter/dist/redux-bug-reporter.css'
import submitFn from 'redux-bug-reporter/lib/integrations/console'


const images = {
  me: require("./images/me.jpg"),
  functional: require("./images/functional.jpg"),
  haxl: require("./images/haxl.jpg"),
  propositions: require("./images/propositions.jpg"),
  wadler: require("./images/wadler.jpg"),
  fault: require("./images/fault.png"),
};

preloader(images);

// Import theme
import createTheme from "@jimmyhmiller/spectacle/lib/themes/default";


require("normalize.css");
require("@jimmyhmiller/spectacle/lib/themes/default/index.css");




const theme = createTheme({
  base03: "#002b36",
  base02: "#073642",
  base01: "#586e75",
  base00: "#657b83",
  base0: "#839496",
  base1: "#93a1a1",
  base2: "#eee8d5",
  base3: "#fdf6e3",
  yellow: "#b58900",
  orange: "#cb4b16",
  red: "#dc322f",
  magenta: "#d33682",
  violet: "#6c71c4",
  blue: "#268bd2",
  cyan: "#2aa198",
  green: "#859900",
  white: "#ffffff",
});


// Spectacle needs a ref
class Dark extends React.Component {
  render() {
    const { children, ...rest } = this.props;
    return (
      <Slide bgColor="base03" {...rest}>
        {children}
      </Slide>
    )
  }
}

// Spectacle needs a ref
const withSlide = Comp => class WithSlide extends React.Component {
  render() {
    const { noSlide=false, maxWidth="80%", slide: Slide = Dark, ...props } = this.props;
    
    if (noSlide) {
      return <Comp {...props} />
    }
    return (
      <Slide maxWidth={maxWidth}>
        <Comp {...props} />
      </Slide>
    )
  }
}

const detectIndent = source => 
  source ? / +/.exec(source)[0].length : 0;

const removeIndent = (indent, source) =>
  source.split("\n")
        .map(s => s.substring(indent, s.length))
        .join("\n")


const code = (source, printWidth=50,) => {
  const spaces = detectIndent(source);
  return format(removeIndent(spaces, source), {printWidth})
}



const BlankSlide = withSlide(() => {
  return <span />;
})

const Code = withSlide(({ source, lang, title, printWidth }) => {
  return (
    <div>
      <Headline noSlide text={title} />
      <CodePane textSize={20} source={code(source, printWidth)} lang={lang} />
    </div>
  )
})
 
const Point = ({ text, textSize=60 }) => 
  <ListItem textSize={textSize} textColor="base2">
    {text}
  </ListItem>

const Points = withSlide(({ children, title, size, styleContainer }) =>
  <div style={styleContainer}>
    <Headline noSlide size={size} textAlign="left" text={title} />
    <List>
      {children}
    </List>
  </div>
)

const Subtitle = ({ color="blue", size=5, text, ...props }) =>
  <Heading textColor={color} size={size} {...props}>
    {text}
  </Heading>

const Headline = withSlide(({ color="magenta", size=2, text, subtext, subtextSize, textAlign="left", caps=true, subtextColor, image }) =>
  <div>
    <Heading
      textAlign={textAlign}
      size={size} 
      caps={caps}
      lineHeight={1}
      textColor={color}
    >
      {text}
    </Heading>
    {subtext && <Subtitle color={subtextColor} text={subtext} textAlign={textAlign} size={subtextSize} />}
    {image && <Image style={{paddingTop: 20}} height="70vh" src={image} />}
  </div>
)

const TwoColumn = withSlide(({ left, right, title }) =>
  <div>
    <Layout> 
      <Fill>
        <Headline color="cyan" size={4} textAlign="center" noSlide text={title} />
        {left}
      </Fill>
      <Fill>
        {right}
      </Fill>
    </Layout>
  </div>
)

const Presentation = ({ children }) => 
  <Deck theme={theme} controls={false} style={{display: 'none'}} transition={["slide"]} transitionDuration={0} progress="none">
    {children}
  </Deck>

// Redux for the Perplexed

// Redux has risen to be the de facto standard for
// building large applications using React. Unfortunately 
// its popularity has led to a bandwagon effect of people 
// using Redux without understanding it. In this talk we 
// will discuss the why we might want to use Redux, how we 
// can use it, and finally discuss the benefits it brings 
// us.

// We will begin with a basic React app and discuss a 
// problem that occurs almost immediately when beginning 
// with React, where to store state. First we will 
// investigate some resolutions to this problems using 
// just vanilla React. Unfortunately, as our application 
// grows, these refactorings become more and more 
// burdensome and a better solution is needed. Redux was 
// made to fill this hole.

// Having motivated Redux, we will explore its three 
// fundamental concepts: reducers, actions, and the store. 
// We will modify our existing application to take 
// advantage of these features, alleviating the pains we 
// felt before and giving our code a strong organizing 
// principle. Finally, we will show what this refactor 
// enabled; using libraries from the redux ecosystem, we 
// will be able to quickly add undo/redo to our existing 
// code, persist our state to local storage, inspect our 
// program with a time travel debugger, and create truly 
// great bug reports with full playback of user actions.

const buttonStyles = {
  border: 'none', 
  color: "#ffffff", 
  marginTop: 10
}

const ColorButton = ({ color, children, onClick }) =>
  <div>
    <button
      onClick={onClick}
      style={{...buttonStyles, backgroundColor: color}}
    >
      {children}
    </button>
  </div>

class Counter extends Component {

  state = {
    count: 0
  }

  increment = () => {
    this.setState({
      count: this.state.count + 1
    })
  }

  render() {
    const { color } = this.props;
    return (
      <ColorButton 
        color={color}
        onClick={this.increment}
      >
        Count: {this.state.count}
      </ColorButton>
    )
  }
}




const UncontrolledCounter = ({ color, count, onClick }) =>
  <ColorButton color={color} onClick={onClick}>
    Count: {count}
  </ColorButton>;

const INCREMENT = 'INCREMENT';

const increment = ({ color }) => ({
  type: INCREMENT,
  color,
})

const mapStateToProps = (state, { color }) => ({
  count: state.count[color]
})

const mapDispatchToProps = (dispatch, { color }) => ({
  onClick: () => dispatch(increment({ color }))
})

const ConnectedCounter = 
  connect(mapStateToProps, mapDispatchToProps)(UncontrolledCounter)



const init = { blue: 0, red: 0, green: 0, purple: 0 }

const colorReducer = (state=init, action) => {
  switch (action.type) {
    case INCREMENT:
      return {
        ...state,
        [action.color]: state[action.color] + 1,
      }
    default:
      return state
  }
}

const store = createStore(combineReducers({count: colorReducer}))

export default () =>
  <Presentation>

    <Headline
      color="green"
      textAlign="left"
      text="Javascript has a bad name" />

    <Headline
      color="blue"
      textAlign="left"
      text="Modern Javascript is all functional" />

    <Headline
      caps={false}
      textAlign="center"
      text="UI = f(state)" />

    <Headline
      size={4}
      caps={false}
      textAlign="center"
      text="state = f(previous_state, event)" />

    <Headline color="blue" text="Redux" />

    <Points title="Concepts">
      <Point text="Actions" />
      <Point text="Reducers" />
      <Point text="Store" />
    </Points>

    <Code
      title="Actions"
      lang="javascript"
      source={`
        const INCREMENT = 'INCREMENT';

        const increment = () => ({
          type: INCREMENT
        })
      `}
    />

    <Code
      title="Actions"
      lang="javascript"
      source={`
        const INCREMENT = 'INCREMENT';

        const increment = ({ color }) => ({
          type: INCREMENT,
          color,
        })
      `}
    />

    <Code
      title="Reducer"
      lang="javascript"
      source={`
        const counterReducer = (state=0, action) => {
          switch (action.type) {
            case INCREMENT:
              return state + 1
            default:
              return state
          }
        }
      `}
    />

    <Code
      title="Reducer"
      lang="javascript"
      source={`
        const init = { blue: 0, red: 0, green: 0, purple: 0 }

        const colorReducer = (state=init, action) => {
          switch (action.type) {
            case INCREMENT:
              return {
                ...state,
                [action.color]: state[action.color] + 1,
              }
            default:
              return state
          }
        }
      `}
      printWidth={65}
    />

    <Code
      title="Store"
      lang="javascript"
      source={`
        import { createStore } from 'redux';
        import { combineReducers } from 'redux';
        import counterReducer from 'src/counterReducer';
        import { increment } from 'src/actions'

        const store = createStore(combineReducers({count: colorReducer}))

        store.subscribe(() =>
          console.log(store.getState())
        )

        store.dispatch(increment({ color: 'blue'}))
        // 1
      `}
    />

    <Headline
      color="cyan"
      text="Redux is a clojure atom but with events" />

    <Slide bgColor="base03" maxWidth={2000}>
      <ComponentPlayground
        theme="light"
        code={code(`

        const Body = () => 
          <div>
            <Counter color="red" />
            <Counter color="green" />
            <Counter color="blue" />
          </div>
        
        const store = createStore(combineReducers({count: colorReducer}))

        render(
          <Provider store={store}>
            <Body />
          </Provider>, mountNode);
        `)}
        scope={{React, Counter: ConnectedCounter, Provider, createStore, colorReducer, combineReducers }}
      />
    </Slide>

    <Slide bgColor="base03" maxWidth={2000}>
      <ComponentPlayground
        theme="light"
        code={code(`

          const BasicCounter = ({ color, count, onClick }) =>
            <ColorButton color={color} onClick={onClick}>
              Count: {count}
            </ColorButton>

          const mapStateToProps = (state, { color }) => ({
            count: state.count[color]
          })

          const mapDispatchToProps = (dispatch, { color }) => ({
            onClick: () => dispatch(increment({ color }))
          })

          const Counter = 
            connect(mapStateToProps, mapDispatchToProps)(BasicCounter)


          render(
            <Provider store={store}>
              <Counter color="blue" />
            </Provider>, mountNode);
        `, 60)}
        scope={{React, Provider, createStore, colorReducer, store, ColorButton, connect, increment }}
      />
    </Slide>

    <Slide bgColor="base03" maxWidth={2000}>
      <ComponentPlayground
        theme="light"
        code={code(`

          const getState = (state) => ({ state })

          const ConditionalRender = connect(getState)(({ pred, state, children }) =>
            pred(state) && children
          )

          const showPurple = (state) => 
            state.count.blue === state.count.red && state.count.red !== 0 

          const Body = () =>
            <div>
              <Counter color="red" />
              <Counter color="green" />
              <Counter color="blue" />
              <ConditionalRender pred={showPurple}>
                <Counter color="purple" />
              </ConditionalRender>
            </div>

          render(
            <Provider store={store}>
              <Body />
            </Provider>, mountNode);
        `, 60)}
        scope={{React, Provider, createStore, colorReducer, store, ColorButton, connect, increment, Counter: ConnectedCounter }}
      />
    </Slide>

    <Points title="What Redux Offers">
      <Point text="Single Source of Truth" />
      <Point text="Isolated Business Logic" />
      <Point text="Immutable Update Model" />
      <Point text="Middleware" />
      <Point text="Ecosystem" />
    </Points>

    <Slide bgColor="base03" maxWidth={2000}>
      <ComponentPlayground
        theme="light"
        code={code(`

        // import undoable from 'redux-existing-undo';
        // import { ActionCreators as doers } from 'redux-existing-undo';

        const Body = ({ undo, redo }) => 
          <div>
            <button onClick={undo}>Undo</button>
            <button onClick={redo}>Redo</button>
            <Counter color="red" />
            <Counter color="green" />
            <Counter color="blue" />
          </div>

        const UndoableBody = connect(null, doers)(Body)
        
        const store = createStore(combineReducers({
          count: undoable(colorReducer),
        }))

        render(
          <Provider store={store}>
            <UndoableBody />
          </Provider>, mountNode);
        `)}
        scope={{React, Counter: ConnectedCounter, Provider, createStore, colorReducer, combineReducers, undoable, doers: ActionCreators, connect }}
      />
    </Slide>

    <Slide bgColor="base03" maxWidth={2000}>
      <ComponentPlayground
        theme="light"
        code={code(`

        const Body = () => 
          <div>
            <Counter color="red" />
            <Counter color="green" />
            <Counter color="blue" />
          </div>
        
        const store = createStore(combineReducers({
          count: colorReducer
        }), window.__REDUX_DEVTOOLS_EXTENSION__ && window.__REDUX_DEVTOOLS_EXTENSION__())

        render(
          <Provider store={store}>
            <Body />
          </Provider>, mountNode);
        `)}
        scope={{React, Counter: ConnectedCounter, Provider, createStore, colorReducer, combineReducers }}
      />
    </Slide>

    <Slide bgColor="base03" maxWidth={2000}>
      <ComponentPlayground
        theme="light"
        code={code(`

        // import { persistStore, autoRehydrate } from 'redux-persist'

        const Body = () => 
          <div>
            <Counter color="red" />
            <Counter color="green" />
            <Counter color="blue" />
          </div>
        
        const store = createStore(combineReducers({
          count: colorReducer
        }), autoRehydrate())

        persistStore(store)

        render(
          <Provider store={store}>
            <Body />
          </Provider>, mountNode);
        `)}
        scope={{React, Counter: ConnectedCounter, Provider, createStore, colorReducer, combineReducers, persistStore, autoRehydrate }}
      />
    </Slide>

    <Slide bgColor="base03" maxWidth={2000}>
      <ComponentPlayground
        theme="light"
        code={code(`

        // import {storeEnhancer} from 'redux-bug-reporter'
        // import ReduxBugReporter from 'redux-bug-reporter'
        // import 'redux-bug-reporter/dist/redux-bug-reporter.css'
        // import submitFn from 'redux-bug-reporter/lib/integrations/console'

        const Body = () => 
          <div>
            <ReduxBugReporter submit={submitFn} projectName='example' />
            <Counter color="red" />
            <Counter color="green" />
            <Counter color="blue" />
          </div>
        
        const store = createStore(combineReducers({
          count: colorReducer
        }), storeEnhancer)

        render(
          <Provider store={store}>
            <Body />
          </Provider>, mountNode);
        `)}
        scope={{React, Counter: ConnectedCounter, Provider, createStore, colorReducer, combineReducers, persistStore, autoRehydrate, storeEnhancer, ReduxBugReporter, submitFn }}
      />
    </Slide>

    <Points title="Rich Ecosystem">
      <Point text="Redux-Thunk" />
      <Point text="Redux-Saga" />
      <Point text="Normalizr" />
      <Point text="Redux-Form" />
    </Points>




    <BlankSlide />




  </Presentation>


