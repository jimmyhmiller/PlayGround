import React, { useState, useEffect, useReducer, useCallback, useMemo } from "react";
import Head from "next/head";
import { useDebounce } from 'use-debounce';

// Fast refresh was too buggy.
// I need to focus on some code clean up
// but probably should make delete and example first
// Should you be allowed to delete referenced components?

// Without fast refresh, my hooks lose their state
// I could probably fix this on my own?
// I could pass my own hook and save off their state and restore it.
// Just do this for each component and I would be fine.


import {
  Editor,
  renderElementAsync
} from "react-live";

const extractComponents = (code) => {
  const compRegex = /<([A-Z][a-zA-Z0-9]*).*\/?>?/g
  const propsRegex = /( [A-Za-z]+)=?/g
  return [...code.matchAll(compRegex)]
    .map(x => ({
      name: x[1],
      props: [...x[0].matchAll(propsRegex)].map(x => x[1])
    }))
 }

const makeComponent = ({ name, code, props }) => {
  const [pre, post] = splitCodeSections(code)
  const hooks = post ? pre : "";
  const result = post !== undefined ? post : pre;
  // Need to figure out hook signatures
  return `
  const ${name} = (${props && props.length > 0  ? "{ " + props.join(", ") + " }" : "props"}) => {
    ${hooks}
    return <>${result}</>
  }
  `
}

const makeComponents = (components) =>
  Object.values(components)
    .filter(c => c.type === "component")
    .map(makeComponent)
    .join("\n");

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

const ReducerEditor = ({ code, actionType, setActions }) => (
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

const filterObject = (object, f) => {
  return Object.fromEntries(Object.entries(object).filter(([k, v]) => f(k, v)))
}

const ReduxIde = ({ initialComponents, initialActions }) => {

  const [components, setComponents] = useState(initialComponents || {Main: {code: "Hello World", name: "Main",  type: "component"}});
  const [actions, setActions] = useState(initialActions || {});

  useEffect(() => {
    const code = extractAllCode(components)
    const stateValue = constructState(code, components["State"] && components["State"]["code"]);
    setAppState(stateValue)
  }, [])


  const reducers = useMemo(() => {
    return Object.values(actions).reduce((reducers, {actionType, code}) => {
      try {
        const reducer = eval(code);
        const discriminatingReducer = (f) => (state, action) => {
          if (action.type !== actionType) {
            return f(state, action)
          }
          return reducer(state, action)
        }
        return {
            ...reducers,
            [actionType]: discriminatingReducer
          }
      }
      catch (e) {
         console.error(e, actionType)
        return reducers
      }
    }, {})

  }, [actions])


  const reducer = useMemo(() => {
    return Object.values(reducers).reduce((f, g) => g(f), defaultReducer)
  }, [reducers]);

  const actionCreators = useMemo(() => {
    return Object.values(actions).reduce((actionObj, {actionType}) => ({
      ...actionObj,
      [actionType]: (args) => dispatch({type: actionType, ...args})
    }), {})
  }, [actions])

  const [appState, dispatch] = useReducer(reducer, {});


  const [debouncedComponents] = useDebounce(components, 200)
  const [Element, setElement] = useState(() => () => null);

  const setAppState = (stateValue) => dispatch({type: "SET_STATE", payload: stateValue})



  const snapshotState = useCallback(() => {
    console.log(JSON.stringify({ components, actions }))
  }, [components])




  useEffect(() => {
    // Maybe split some things out here?
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
      ...filterObject(actions, (_,  {code, actionType, props}) => code !== defaultReducerCode(appState, props) || inCodeActions.has(actionType)),
      ...additionalActions,
    }))
    const isValidState = Object.keys(stateValue).length > 0 && (!components["State"] || JSON.stringify(stateValue, null, 1) !== components["State"]["code"])

    const stateComponent = {
      code: JSON.stringify(stateValue, null, 1),
      name: "State",
      type: "state",
    }

    if (isValidState) {
      setAppState(stateValue)
      setComponents(components => ({
        ...components,
        State: stateComponent
      }))
    }

    // still ugly
    const comps = extractComponents(code);
    const inComps = new Set(Object.values(comps).map(c => c.name).concat("Main"))
    const newComps = comps.filter(c => components[c.name] === undefined);
    const modifiedComps = comps.filter(c => components[c.name] && JSON.stringify(components[c.name].props) !== JSON.stringify(c.props))
    const notMentioned = Object.entries(components).filter(([_, {name, code, type}]) => !inComps.has(name) && type === "component");

    if (newComps.length > 0 || notMentioned.length > 0 || modifiedComps.length > 0) {
      setComponents(components => ({
        ...filterObject(components, (_, {name, code}) => (code !== name && code !== "") || inComps.has(name)),
        ...Object.fromEntries(newComps.map(c => [c.name, {code: c.name, name: c.name, type: "component", props: c.props}])),
        ...Object.fromEntries(modifiedComps.map(c => [c.name, {...components[c.name], ...c}])),
      }))
    }
   

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
          setElement((_) => elem)
         },
          e => { 
            console.error(e, "error rendering");
            setElement((_) => Element)
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
          a {
            color: #fff;
            text-decoration: none;
            cursor: pointer;
          }
        `}
      </style>
      <Head>
        <title>Redux Like Editor</title>
      </Head>

      <div style={{display: "flex", flexDirection: "row"}}>
        {false && <button onClick={_ => snapshotState()}>Snapshot</button>}
        <div style={{width: "45vw", height: "95vh", overflow: "scroll"}}>

          {Object.values(components).filter(c => c.type === "component").map(({ code, name, props }) =>
            <ComponentEditor name={name} code={code} setComponents={setComponents} props={props} />
          )}

          {Object.values(actions).map(({ actionType, code }) =>
            <ReducerEditor
              actionType={actionType}
              code={code}
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

export default ReduxIde;