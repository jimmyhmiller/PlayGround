import React, { useState, useEffect, useReducer, useCallback } from "react";
import Head from "next/head";
import { useDebounce } from 'use-debounce';


// This fast refresh stuff is cool. But might be too error prone.
// Might need to change back to the hackier way.


// This is the beginning of geting fast refresh to work.
// If I paste that second if statement below and 
// change Thing to return something else,
// then I can actually get a fast refresh working in the browser
// by calling window.refresh()
// I should be able to get this hooked up to my render stuff
// and then I can fast refresh my components and actually
// have them keep hook state and things like that.
if (process.env.NODE_ENV !== 'production' && typeof window !== 'undefined') {
  const runtime = require('react-refresh/runtime');
  const { stopReportingRuntimeErrors } = require("react-error-overlay");
  stopReportingRuntimeErrors();
  window.runtime = runtime;
  window.refresh = runtime.performReactRefresh;
  runtime.injectIntoGlobalHook(window);
}

import {
  Editor,
  renderElementAsync
} from "react-live";

const extractComponents = (code) => {
  const compRegex = /<([A-Z][a-zA-Z0-9]+).*\/?>?/g
  const propsRegex = /([A-Za-z]+)=/g
  return [...code.matchAll(compRegex)]
    .map(x => ({
      name: x[1],
      props: [...x[0].matchAll(propsRegex)].map(x => x[1])
    }))
 }
 const genSym = (() => {
   let i = 0;
   return (prefix) => { i += 1; return `${prefix || "var"}_${i}`}
 })()

const makeComponent = ({ name, code, props }) => {
  const [pre, post] = splitCodeSections(code)
  const hooks = post ? pre : "";
  const result = post !== undefined ? post : pre;
  // Need to figure out hook signatures
  const sig = genSym("sig")
  return `
  const ${sig} = window.runtime.createSignatureFunctionForTransform();
  const ${name} = (${props && props.length > 0  ? "{ " + props.join(", ") + " }" : "props"}) => {
    ${sig}();
    ${hooks}
    return <>${result}</>
  }
  ${sig}(${name}, "", null, null)
  window.runtime.register(${name}, "jimmy/${name}")
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

// Allow deletion
// Allow loading of examples
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
// Think about apis
// Show live data


const GenericEditor = ({ code, onValueChange, name, backgroundColor }) => (
  <div
    style={{
      background: `${backgroundColor} none repeat scroll 0% 0%`,
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
      onValueChange={onValueChange}
    />
  </div>

)

const ComponentEditor = ({ name, code, setComponents, props }) => (
  <GenericEditor
    name={<>{name}{props && props.length > 0 && `({ ${props && props.join(", ")} })`}</>}
    code={code}
    backgroundColor="rgb(50, 42, 56)"
    onValueChange={value => {
      setComponents(comps => ({
        ...comps,
        [name]: {...(comps[name] || {}), code: value, name,  }
      }));
    }}
  />
);


const StateEditor = ({ name, code, components, setComponents, setAppState }) => (
  <GenericEditor
    name={name}
    code={code}
    backgroundColor={"rgb(56, 42, 42)"}
    onValueChange={
      value => {
        const code = extractAllCode(components)
        const stateValue = constructState(code, value);
        const formattedCode = JSON.stringify(stateValue, null, 1);

        // Basically this is checking if the state is valid.
        if (Object.keys(stateValue).length > 0) {
          setAppState(stateValue)
        }

        setComponents((components) => ({
          ...components,
          State: {
            // If there are not state values, just use the string value passed in
            code: Object.keys(stateValue).length > 0 ? formattedCode : value,
            name: "State",
            type: "state",
          },
        }))
      }
    }
  />
);


const codeToDestructure = (props, placeholder="_", rest="") => {
  return props && props.length > 0  ? "{ " + props.join(", ") + rest + " }" : placeholder
}

const defaultReducerCode = (appState, props) =>
`(${codeToDestructure(Object.keys(appState), "state", ", ...state")}, ${codeToDestructure(props)}) => ({
  ...state,
})`

const ReducerEditor = ({ code, setReducers, actionType, setActions }) => (
  <GenericEditor
    code={code}
    backgroundColor={"rgb(42, 47, 56)"}
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
      name={actionType}
    />
);

const extractState = code => {
  return Object.fromEntries(
    [...code.matchAll(/State\.([a-zA-Z_0-9]+)/g)].map(x => [x[1], null])
  );
};

const constructState = (componentCode, stateComponentCode = "{}") => {
  try {
    return {
      ...extractState(componentCode),
      ...Object.fromEntries(
        Object.entries(JSON.parse(stateComponentCode)).filter(
          x => x[1] !== null
        )
      )
    }
  } catch (e) {
    return {};
  }
};

const extractActions = (appState, code) => {
  const propsRegex = /([a-zA-Z]+)(:|,| |})?/
  return [...code.matchAll(/Actions\.([a-zA-Z_0-9]+)(\(.*\)?)?/g)]
    .map(([_, action, args]) => {
      const props = (args||"")
        .split(",")
        .map(prop => prop.match(propsRegex))
        .filter(x => x)
        .map(prop => prop[1])

      return {
        actionType: action,
        props: props,
        code: defaultReducerCode(appState, props),
      }
    })
};


const extractAllCode = (components) => {
  return Object.values(components)
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
  const [debouncedComponents] = useDebounce(components, 0)
  const [Element, setElement] = useState(() => () => null);
  const [firstRender, setFirstRender] = useState(true);

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

    const extractedActions = extractActions(appState, code);

    const additionalActions = extractedActions
      .filter(action => !actions[action.actionType] || actions[action.actionType].props && actions[action.actionType].props !== action.props)
      .reduce((obj, action) => ({
        ...obj,
        [action.actionType]: action,
      }), {})

    const inCodeActions = new Set(Object.values(extractedActions).map(a => a.actionType))

    setActions((actions) => ({
      ...Object.fromEntries(Object.entries(actions).filter(([_, {code, actionType, props}]) => code !== defaultReducerCode(appState, props) || inCodeActions.has(actionType))),
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

    // This is terrible. Need to fix up
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
      renderElementAsync({ code: wrapCode(components), scope: {React, useState, useEffect, State: appState, Actions: actionCreators, builtin: { Editor, renderElementAsync }} },
        // Probably a race condition with this error?
        // But in general, it should render and then as it is rendering,
        // if there is an error this would be called.
        // It would be better if this library would only call on a fully successful render.
        // Not sure how to accomplish that at the moment.
        // Tried a useEffect, but that still got called even when its children errored.
         (elem) => {
           if (firstRender) {
              setElement((_) => elem)
              setFirstRender(false)
            } else {
              setTimeout(() => window.refresh(), 0);
            }
           
         },
          e => { 
            console.error(e, "error rendering");
            // Need to figure out how to make fast refresh use the old code here.
            window.refresh();
          });
    } catch (e) {
      console.error(e, "error in the rendering function")
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
        <title>Redux Like Editor</title>
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