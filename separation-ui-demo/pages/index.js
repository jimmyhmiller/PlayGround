import Head from 'next/head'
import React, { useContext, useState, useEffect } from 'react';


const buttonContext = React.createContext({})


const Provider = ({ children }) => {
  const props = {text: "Click Me"};
  const [context, setContext] = useState({});
  const register = ({ name, props, onInteract }) => {
    setContext(context => ({
      ...context,
      [name]: {
        ...(context.name || {}),
        onInteract: onInteract || (context.name && context.name.interact),
        props,
        registered: onInteract || (context.name && context.name.interact),
        setProps: (props) => setContext(context => ({
          ...context,  
          [name]: {
            ...context[name],
            props,
          }
        })),
      }
    }))
  }
  return (
      <buttonContext.Provider value={{...context, register}}>
        {children}
      </buttonContext.Provider>
  )
}

const useInteract = (name, onInteract, deps) => {
  const {register} = useContext(buttonContext);
  useEffect(() => {
    register({name, onInteract})
  }, [])
  
}

const useButtonState = ({ name, compProps }) => {
  const {[name]: things, register} = useContext(buttonContext);
  const { onInteract, setProps, props, registered } = things || {};
  const [clicked, setClicked] = useState(false);
  const [hover, setHover] = useState(false);
  useEffect(() => {
    register({name, ...compProps});
  }, [])
  useEffect(() => {
    if (!onInteract) {
      return;
    }
    const result = onInteract({ clicked, setProps, hover, props });
    if (result) {
      setProps(result)
    }
    if (clicked) {
      setClicked(false)
    }
  }, [clicked, hover])

  useEffect(() => {
    if (registered) {
      const result = onInteract({ clicked, setProps, hover, props });
      if (result) {
        setProps(result)
      }
    }
    
  }, [registered]);

  return {
    clicked,
    setClicked,
    hover,
    setHover
  }
}

const Button = ({ name, ...compProps }) => {
  const { setClicked, setHover } = useButtonState({ name, compProps });
  const { [name]: things} = useContext(buttonContext);
  const { props } = things || {props: {}};
  return (
    <button
      {...props}
      onClick={_ => setClicked(true)}
      onMouseEnter={_ => setHover(true)}
      onMouseLeave={_ => setHover(false)}>
      {props && props.text || compProps && compProps.text || ""}
    </button>
   )
}


const NestedComp = () => <Button name="test" text="Click Me" />




const BlueContainer = ({ children }) => {
  useInteract('test', ({ clicked, hover, props }) => {
    if (clicked) {
      console.log("test");
    }
    return {
      ...props,
      style: {
        backgroundColor: hover ? "gray" : "blue"
      }
    }
  })
  return (
    <div>
      Blue Hello World
      {children}
    </div>
  )
}

const RedContainer = ({ children }) => {
  useInteract('test', ({ clicked, hover, props }) => {
    if (clicked) {
      console.log("test!!!!!");
    }
    return {
      ...props,
      style: {
        backgroundColor: hover ? "gray" : "red"
      }
    }
  })
  return (
    <div>
      Red Hello World
      {children}
    </div>
  )
}



const Home = () => {
  return (
  <Provider>
   <RedContainer>
     <NestedComp />
    </RedContainer>
  </Provider>
  )
}

export default Home
