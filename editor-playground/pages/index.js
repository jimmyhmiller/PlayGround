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
    const [state, setState] = useState(undefined);
    ${hooks}
    return <>${result}</>
  }
  `
}

const makeComponents = (components) => 
  Object.values(components).map(makeComponent).join("\n");

// Hack to play with the concept
const splitCodeSections = (code) => code.split("--")

const wrapCode = (components) => {
  return `
  ${makeComponents(components)}
  render(Main);
  `
}


// Extract out components to make code below better
// Label each editor
// Allow deletion
// Only create if finished
// Work on state management (different color editor)
// Use actually parser
// Debounce input
// Destructure props and show them
// Prettier
// Iframe/layout mechanism for proper setup?
// Worker task?
// Keep hooks state?


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
          [name]: { code: value, name }
        }));
      }}
    />
  </div>
);



const Home = () => {

  const [components, setComponents] = useState({Main: {code: "Hello World", name: "Main"}});
  const [debouncedComponents] = useDebounce(components, 200)
  const [Element, setElement] = useState(() => () => null);

  useEffect(() => {
    const comps = extractComponents(Object.values(debouncedComponents).map(c => c.code).join("\n"));
    comps.forEach(c => {
      if (!components[c]) {
        setComponents((components) => ({
          ...components,
          [c]: {code: c, name: c},
        }))
      }
    })

  }, [debouncedComponents])

  useEffect(() => {
    try {
      renderElementAsync({ code: wrapCode(components), scope: {React, useState} }, 
        (elem) => setElement((_) => elem),  e => console.log(e));
    } catch (e) {
      console.error(e)
    }
  }, [components])

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

          {Object.values(components).map(({ code, name }) => 
            <ComponentEditor name={name} code={code} setComponents={setComponents} />
          )}
          
        </div>
        <div style={{width: "45vw", height: "95vh", padding: 20}}>
          <Element />
        </div>
      </div>


    </div>
  );
};

export default Home;