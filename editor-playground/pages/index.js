import React, { useState, useEffect } from "react";
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

const extractComponents = (code) =>
  [...code.matchAll(/<([A-Z][a-zA-Z0-9]+).*\/?>/g)].map(x => x[1])

const makeComponent = ({ name, code}) => {
  const [pre, post] = splitCodeSections(code)
  const hooks = post ? pre : "";
  const result = hooks ? post : pre;
  return `
  const ${name} = (props) => {
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
// Work on state management (different color editor)
// Use actually parser
// Destructure props and show them
// Prettier
// Iframe/layout mechanism for proper setup?
// Worker task?
// Keep hooks state?
// Dispatch
// Better place to store state


const ComponentEditor = ({ name, code, setComponents }) => (
  <div
    style={{
      background: "rgb(50, 42, 56) none repeat scroll 0% 0%",
      marginTop: 10,
      borderRadius: 5
    }}
  >
    <div style={{padding:10, borderBottom: "2px solid rgb(50, 42, 56)", filter: "brightness(80%)"}}>
      {name}
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

const extractAllCode = (components) => {
  return Object.values(
        components
      )
      .filter(c => c.type === "component")
      .map(c => c.code)
      .join("\n");
}

const Home = () => {
  const [appState, setAppState] = useState({});
  const [components, setComponents] = useState({Main: {code: "Hello World", name: "Main",  type: "component"}});
  const [debouncedComponents] = useDebounce(components, 200)
  const [Element, setElement] = useState(() => () => null);

  useEffect(() => {
    const code = extractAllCode(debouncedComponents)

    const stateValue = constructState(code, components["State"] && components["State"]["code"]);
    const formattedCode = JSON.stringify(stateValue, null, 1);

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
    comps.forEach(c => {
      if (!components[c]) {
        setComponents((components) => ({
          ...components,
          [c]: {code: c, name: c, type: "component"},
        }))
      }
    })

  }, [components])

  useEffect(() => {
    try {
      renderElementAsync({ code: wrapCode(components), scope: {React, useState, State: appState} }, 
        (elem) => setElement((_) => elem), e => console.error(e));
    } catch (e) {
      console.error(e)
    }
  }, [debouncedComponents])

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

        <div style={{width: "45vw", height: "95vh"}}>

          {Object.values(components).filter(c => c.type === "component").map(({ code, name }) => 
            <ComponentEditor name={name} code={code} setComponents={setComponents} />
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