const createStore = (reducer, initialState, enhancer) => {

  if (enhancer) {
    return enhancer(createStore)(reducer, initialState)
  }

  let state = reducer(initialState, {});
  let subscribers = []

  return {
    getState: () => state,
    dispatch: (action) => {
      state = reducer(state, action)
      subscribers.forEach(f => f())
      return action;
    },
    subscribe: (f) => {
      subscribers.push(f);
      return () => {
        const index = subscribers.indexOf(f);
        subscribers.splice(index, 1)
        return undefined
      }
    }
  }
}

const compose = (f, g) => (...args) => f(g(...args))

const applyMiddleware = (...middlewares) => (createStore) => (reducer, initialState) => {
  const store = createStore(reducer, initialState);
  const newDispatch = middlewares
    .map(m => m(store))
    .reduce(compose)(store.dispatch)

  return {
    ...store,
    dispatch: newDispatch
  }
}














// const loggingMiddleware = (num) => ({ getState }) => next => action => {
//   console.log(`before action ${num}`, getState());
//   const result = next(action);
//   console.log("after action", getState());
//   return result;
// }


// const reducer = (state=0, action) => {
//   switch (action.type) {
//     case "INCREMENT": {
//       return state + 1;
//     }
//     case "DECREMENT": {
//       return state - 1;
//     }
//     default: {
//       return state;
//     }
//   }
// }


// const middleware = applyMiddleware(
//   loggingMiddleware(1), 
//   loggingMiddleware(2),
// )

// const store = createStore(reducer, 0, middleware)


// store.dispatch({type: "INCREMENT"})
// store.dispatch({type: "INCREMENT"})
// store.dispatch({type: "INCREMENT"})
// store.dispatch({type: "INCREMENT"})
// store.dispatch({type: "INCREMENT"})
// store.dispatch({type: "DECREMENT"})


