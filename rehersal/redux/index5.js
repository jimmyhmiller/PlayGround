const createStore = (reducer, initialState, enhancer) => {
  if (enhancer) {
    return enhancer(createStore)(reducer, initialState)
  }
  let state = reducer(initialState, {})
  const listeners = []

  return {
    getState: () => state,
    dispatch: (action) => {
      state = reducer(state, action)
      listeners.forEach(l => l());
      return action;
    },
    subscribe: (listener) => {
      listeners.push(listener);
      return () => {
        const index = listeners.indexOf(listener);
        listeners.splice(index, 1);
      }
    }
  }
}


// Dispatch :: Action -> Action

// loggingMiddleware :: Store -> Dispatch -> Dispatch

const loggingMiddleware = ({ getState }) => (next) => (action) => {
  console.log("State before call", getState());
  const result = next(action);
  console.log("State after call", getState());
  return result;
}


const counterReducer = (state=0, action) => {
  switch (action.type) {
    case "INCREMENT": {
      return state + 1;
    }
    case "DECREMENT": {
      return state - 1;
    }
  }
  return state;
}
const applyMiddleware = (middleware) => (createStore) => (...args) => {
  const store = createStore(...args);
  const dispatch = middleware(store)(store.dispatch);
  return {
    ...store,
    dispatch,
  }
}


let store = createStore(counterReducer, 0, applyMiddleware(loggingMiddleware))


store.dispatch({type: "INCREMENT"})
store.dispatch({type: "INCREMENT"})
store.dispatch({type: "INCREMENT"})
store.dispatch({type: "INCREMENT"})
store.dispatch({type: "INCREMENT"})

store.dispatch({type: "INCREMENT"})
store.dispatch({type: "DECREMENT"})


