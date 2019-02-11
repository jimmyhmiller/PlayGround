const createStore = (reducer, initialState, enhancer) => {
  if (enhancer) {
    return enhancer(createStore)(reducer, initialState)
  }
  let state = reducer(initialState, {})
  let listeners = [] 

  return {
    getState: () => state,
    dispatch: (action) => {
      state = reducer(state, action)
      listeners.forEach(f => f());
      return action
    },
    subscribe: (f) => {
      listeners.push(f);
      return () => {
        const index = listeners.indexOf(f);
        listeners.splice(index, 1)
      }
    }
  }
}

const compose = (f, g) => (...args) => f(g(...args))

const applyMiddleware = (...middlewares) => (createStore) => (...args) => {

  const store = createStore(...args);
  const dispatch = middlewares
    .map(m => m(store))
    .reduce(compose)(store.dispatch)
  return {
    ...store,
    dispatch,
  };
}



// const loggingMiddleware = ({ getState }) => next => action => {
//   console.log("before action", getState());
//   const result = next(action);
//   console.log("after action", getState());
//   return result;
// }


// const reducer = (state=0, action) => {
//     switch (action.type) {
//         case "INCREMENT": {
//             return state + 1;
//         }
//         case "DECREMENT": {
//             return state - 1;
//         }
//         default: {
//             return state;
//         }
//     }
// }



// const store = createStore(reducer, 0, applyMiddleware(loggingMiddleware))

// store.subscribe(() => console.log(store.getState()))

// store.dispatch({type: "INCREMENT"})
// store.dispatch({type: "INCREMENT"})









