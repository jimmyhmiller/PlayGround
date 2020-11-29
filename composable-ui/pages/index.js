import React, { useState, useImperativeHandle, forwardRef, useRef, useMemo, useCallback, useEffect } from 'react'
import Head from 'next/head'


const useInput = (initialValue, onChange) => {
  const [value, setValue] = useState(initialValue);
  const onChangeInternal = (value) => {
    if (onChange) {
      onChange(value);
    }
    setValue(value);
  }
  return [value, onChangeInternal];
}

const Input = ({ onChange }) => {
  const [value, setValue] = useInput("", onChange);
  return (
      <div>
        <label>Name: </label>
        <input value={value} onChange={(e) => setValue(e.target.value)} />
      </div>
    )
}

const useValue = (initialValue) => {
  const [value, setValue] = useState(initialValue);
  return [value, setValue];
}


const ComponentBorder = ({ children, selected, hide, name }) =>
    hide ?
    <>{children}</>
     :
    <div style={{padding: 20, border: `solid 1px ${selected ? "red" : "black"}`, width:250, margin: 20}}>
      {children}
    </div>

const Greet = ({ InputComponent, onChange, selected }) => {
  const [value, setValue] = useInput("placeholder", onChange);
  useEffect(() => {
    if (onChange) {
      onChange(<>Hello {value}</>)
    }
  }, [value])

  return (
    <ComponentBorder selected={selected} hide={!!onChange}>
      {InputComponent && <InputComponent onChange={setValue} />}
      {onChange ? null : <p>Hello {value}</p>}
    </ComponentBorder>
  )
}

const Large = ({ InputComponent, onChange, selected }) => {
  const [value, setValue] = useInput("placeholder", onChange);
  useEffect(() => {
    if (onChange) {
      onChange(<h1 style={{display: "inline"}}>{value}</h1>)
    }
  }, [value])

  return (
    <ComponentBorder selected={selected} hide={!!onChange}>
      {InputComponent && <InputComponent onChange={setValue} />}
      {onChange ? null : <h1 style={{display: "inline"}}>{value}</h1>}
    </ComponentBorder>
  )
}



const Blue = ({ InputComponent, onChange, selected }) => {
  const [value, setValue] = useInput("placeholder", onChange);
  useEffect(() => {
    if (onChange) {
      onChange(<span style={{color: "blue"}}>{value}</span>)
    }
  }, [value])

  return (
    <ComponentBorder selected={selected} hide={!!onChange}>
      {InputComponent && <InputComponent onChange={setValue} />}
      {onChange ? null : <span style={{color: "blue"}}>{value}</span>}
    </ComponentBorder>
  )
}

const ComposeOutputs = (Output1, Output2) => ({ InputComponent, onChange, selected }) => {
  const FirstComp = useRef(({onChange}) => <Output1 InputComponent={InputComponent} onChange={onChange} />);
  return (
    <Output2 InputComponent={FirstComp.current} onChange={onChange} selected={selected} />
  )
}

const GreetLarge = ComposeOutputs(ComposeOutputs(Greet, Large), Blue);

const initComponents = {
    greet: {component: Greet}, 
    large: {component: Large}, 
    blue: {component: Blue},
  }

const Home = () => {
  const [components, setComponents] = useState(initComponents)

  const [selected, setSelected] = useState(null);

  const composeUp = ({ key }) => (e) => {
    if (e.metaKey) {
      if (!selected) {
        setSelected(key);
      } else {
        const { [selected]: _, ...comps } = components;
        setComponents({
          ...comps,
          [key]: {component: ComposeOutputs(components[key].component, components[selected].component)},
        })
        setSelected(null)
      }
    }
  }

  return (
    <div>
      <Head>
        <title>Compose Components</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <h1>Compose Components</h1>
      <p>Command click to select components and compose them</p>
      <button onClick={_ => setComponents(initComponents)}>Reset</button>
      {Object.entries(components).map(
        ([key, value]) => (
          <div key={key} onClick={composeUp({key})}>
            <value.component InputComponent={Input} selected={selected === key} />
          </div>
        )
      )} 
    </div>
  )
}

export default Home
