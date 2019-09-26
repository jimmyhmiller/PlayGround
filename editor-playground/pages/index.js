import React, { useState, useEffect, useReducer } from "react";
import Head from "next/head";
import { useDebounce } from 'use-debounce';

import {
  LiveProvider,
  LiveEditor,
  LiveError,
  LivePreview,
  Editor,
  generateElement,
  renderElementAsync
} from "react-live";

// rgb(42, 47, 56)
// rgb(56, 42, 42)

const extractComponents = (code) => {
  const compRegex = /<([A-Z][a-zA-Z0-9]+).*\/?>/g
  const propsRegex = /([A-Za-z]+)=/g
  return [...code.matchAll(compRegex)]
    .map(x => ({
      name: x[1],
      props: [...x[0].matchAll(propsRegex)].map(x => x[1])
    }))
 }

const makeComponent = ({ name, code, props }) => {
  const [pre, post] = splitCodeSections(code)
  const hooks = post ? pre : "";
  const result = hooks ? post : pre;
  return `
  const ${name} = (${props && props.length > 0  ? "{ " + props.join(", ") + " }" : "props"}) => {
    ${hooks}
    return <>${result}</>
  }
  `
}

const makeComponents = (components) => 
  Object.values(components).filter(c => c.type === "component").map(makeComponent).join("\n");

// Hack to play with the concept
const splitCodeSections = (code) => code.split("--")

const wrapCode = (components) => {
  return `
  ${makeComponents(components)}
  render(Main);
  `
}


// Extract out components to make code below better
// Allow deletion
// Only create if finished
// Use actually parser
// Destructure props and show them
// Prettier
// Iframe/layout mechanism for proper setup?
// Worker task?
// Keep hooks state?
// Dispatch
// Better place to store state
// Combine state and reducers into one box?


const ComponentEditor = ({ name, code, setComponents, props }) => (
  <div
    style={{
      background: "rgb(50, 42, 56) none repeat scroll 0% 0%",
      marginTop: 10,
      borderRadius: 5
    }}
  >
    <div style={{padding:10, borderBottom: "2px solid rgb(50, 42, 56)", filter: "brightness(80%)"}}>
      {name}{props && props.length > 0 && `({ ${props && props.join(", ")} })`}
    </div>
    <Editor
      key={name}
      padding={20}
      language="jsx"
      code={code}
      onValueChange={value => {
        setComponents(comps => ({
          ...comps,
          [name]: {...(comps[name] || {}), code: value, name,  }
        }));
      }}
    />
  </div>
);



const StateEditor = ({ name, code, components, setComponents, setAppState }) => (
  <div
    style={{
      background: "rgb(42, 47, 56) none repeat scroll 0% 0%",
      marginTop: 10,
      borderRadius: 5
    }}
  >
    <div style={{padding:10, borderBottom: "2px solid rgb(42, 47, 56)", filter: "brightness(80%)"}}>
      {name}
    </div>
    <Editor
      key={name}
      padding={20}
      language="jsx"
      code={code}
      onValueChange={
        value => {
          const code = extractAllCode(components)
          const stateValue = constructState(code, value);
          const formattedCode = JSON.stringify(stateValue, null, 1);

          if (Object.keys(stateValue).length > 0) {
            setAppState(stateValue)
            setComponents((components) => ({
              ...components,
              State: {
                code: formattedCode,
                name: "State",
                type: "state",
              },
            }))
          } else {
            setComponents((components) => ({
              ...components,
              State: {
                code: value,
                name: "State",
                type: "state",
              }
            }))
          }
        }
      }
    />
  </div>
);

const defaultReducerCode = (props) => `(state, ${props && props.length > 0  ? "{ " + props.join(", ") + " }" : "_"}) => ({
  ...state,

})`

const ReducerEditor = ({ code, setReducers, actionType, setActions }) => (
  <div
    style={{
      background: "rgb(42, 47, 56) none repeat scroll 0% 0%",
      marginTop: 10,
      borderRadius: 5
    }}
  >
    <div style={{padding:10, borderBottom: "2px solid rgb(42, 47, 56)", filter: "brightness(80%)"}}>
      {actionType}
    </div>
    <Editor
      key={actionType}
      padding={20}
      language="jsx"
      code={code}
      onValueChange={
        value => {
          setActions(actions => ({
            ...actions,
            [actionType]: {actionType, code: value}
          }))
          try {
            const reducer = eval(value);
            const discriminatingReducer = (f) => (state, action) => {
              if (action.type !== actionType) {
                return f(state, action)
              }
              return reducer(state, action)
            }
            setReducers(reducers => ({
              ...reducers,
              [actionType]: discriminatingReducer
            }))
          } catch (e) {
            console.error(e)
          }
        }
      }
    />
  </div>
);

const extractState = code => {
  return Object.fromEntries(
    [...code.matchAll(/State\.([a-zA-Z_0-9]+)/g)].map(x => [x[1], null])
  );
};

const constructState = (componentCode, stateComponentCode = "{}") => {
  try {
    return Object.assign(
      {},
      extractState(componentCode),
      Object.fromEntries(
        Object.entries(JSON.parse(stateComponentCode)).filter(
          x => x[1] !== null
        )
      )
    );
  } catch (e) {
    return {};
  }
};

const extractActions = code => {
  const propsRegex = /([a-z]+)(:|,| |})/g
  return [...code.matchAll(/Actions\.([a-zA-Z_0-9]+)(\(.*\))?/g)]
    .map(x => {
      const props = [...x[0].matchAll(propsRegex)].map(x => x[1]);
      console.log(x[0], props)
      return {
        actionType: x[1],
        props: props,
        code: defaultReducerCode(props),
      }
    })
};


const extractAllCode = (components) => {
  return Object.values(
        components
      )
      .filter(c => c.type === "component")
      .map(c => c.code)
      .join("\n");
}

const defaultReducer = (state, action={}) => {
  if (action.type === "SET_STATE") {
    return action.payload;
  }
  return state;
}

const Home = () => {
  const [reducers, setReducers] = useState({});
  const [reducer, setReducer] = useState(() => defaultReducer);
  const [appState, dispatch] = useReducer(reducer, {});
  const [actions, setActions] = useState([]);
  const [actionCreators, setActionCreators] = useState();
  const [components, setComponents] = useState({Main: {code: "Hello World", name: "Main",  type: "component"}});
  const [debouncedComponents] = useDebounce(components, 200)
  const [Element, setElement] = useState(() => () => null);

  const setAppState = (stateValue) => dispatch({type: "SET_STATE", payload: stateValue})

  useEffect(() => {
    setReducer(() => Object.values(reducers).reduce((f, g) => g(f), defaultReducer))
  }, [reducers])

  useEffect(() => {
    setActionCreators(
      Object.values(actions).reduce((actionObj, {actionType}) => ({
        ...actionObj,
        [actionType]: (args) => dispatch({type: actionType, ...args})
      }), {})
    )
  }, [actions])

  useEffect(() => {
    const code = extractAllCode(components)

    const stateValue = constructState(code, components["State"] && components["State"]["code"]);
    const formattedCode = JSON.stringify(stateValue, null, 1);

    const extractedActions = extractActions(code);

    const additionalActions = extractedActions
      .filter(action => !actions[action.actionType] || actions[action.actionType] && actions[action.actionType].props && actions[action.actionType].props.length < action.props.length)
      .reduce((obj, action) => ({
        ...obj,
        [action.actionType]: action,
      }), {})

    const inCodeActions = new Set(Object.values(extractedActions).map(a => a.actionType))

    setActions((actions) => ({
      ...Object.fromEntries(Object.entries(actions).filter(([_, {code, actionType, props}]) => code !== defaultReducerCode(props) || inCodeActions.has(actionType))),
      ...additionalActions,
    }))

    if (Object.keys(stateValue).length > 0 && (!components["State"] || JSON.stringify(stateValue, null, 1) !== components["State"]["code"])) {
      setAppState(stateValue)
      setComponents((components) => ({
        ...components,
        State: {
          code: JSON.stringify(stateValue, null, 1),
          name: "State",
          type: "state",
        },
      }))
    }

    const comps = extractComponents(code);

    const inComps = new Set(Object.values(comps).map(c => c.name))

    comps.forEach(c => {
      if (!components[c.name]) {
        setComponents((components) => ({
          ...Object.fromEntries(Object.entries(components).filter(([_, {code, name}]) => code !== name || inComps.has(name))),
          [c.name]: {code: c.name, name: c.name, type: "component", props: c.props},
        }))
      }
      if (components[c.name] && components[c.name].props.length < c.props.length) {
          setComponents((components) => ({
          ...components,
          [c.name]: {
            ...components[c.name],
            props: c.props
          },
        }))
      }
    })

  }, [components])

  useEffect(() => {
    try {
      renderElementAsync({ code: wrapCode(components), scope: {React, useState, State: appState, Actions: actionCreators} }, 
        (elem) => setElement((_) => elem), e => console.error(e));
    } catch (e) {
      console.error(e)
    }
  }, [debouncedComponents, appState])

  return (
    <div>
      <style jsx global>
        {`
          body {
            background-color: #363638;
            color: #fff;
            font-family: sans-serif;
          }
        `}
      </style>
      <Head>
        <title>Home</title>
      </Head>

      <div style={{display: "flex", flexDirection: "row"}}>

        <div style={{width: "45vw", height: "95vh", overflow: "scroll"}}>

          {Object.values(components).filter(c => c.type === "component").map(({ code, name, props }) => 
            <ComponentEditor name={name} code={code} setComponents={setComponents} props={props} />
          )}

          {Object.values(actions).map(({ actionType, code }) =>
            <ReducerEditor 
              actionType={actionType} 
              code={code} 
              setReducers={setReducers}
              setActions={setActions} />
          )}

          {components["State"] &&
            <StateEditor 
              name="State" 
              code={components["State"]["code"]}
              components={components} 
              setComponents={setComponents}
              setAppState={setAppState} />
          }
          
        </div>
        <div style={{width: "45vw", height: "95vh", padding: 20}}>
          <Element />
        </div>
      </div>


    </div>
  );
};

export default Home;