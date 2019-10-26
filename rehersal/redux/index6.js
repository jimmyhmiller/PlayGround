const createStore = (reducer, initialState, enhancer) => {
  if (enhancer) {
    return enhancer(createStore)(reducer, initialState);
  }

  let state = reducer(initialState, {});
  const listeners = [];

  return {
    getState: () => state,
    dispatch: (action) => {
      state = reducer(state, action);
      listeners.forEach(listener => listener());
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


const counterReducer = (state=0, action) => {
  switch (action.type) {
    case "INCREMENT": {
      return state + 1
    }
    case "DECREMENT": {
      return state -1
    }
    default: {
      return state;
    }
  }
}

const logMiddleware = (store) => (next) => (action) => {
  console.log(`state before ${store.getState()}`);
  next(action);
  console.log(`state after ${store.getState()}`);
  return action;
}

const applyMiddleware = (applyMiddleware) => (createStore) => (...args) => {
  const store = createStore(...args);
  const dispatch = logMiddleware(store)(store.dispatch);
  return {
    ...store,
    dispatch,
  }
}

const store = createStore(counterReducer, 0, applyMiddleware(logMiddleware))



store.dispatch({type: "INCREMENT"})
store.dispatch({type: "INCREMENT"})
store.dispatch({type: "INCREMENT"})
store.dispatch({type: "DECREMENT"})
store.dispatch({type: "INCREMENT"})
store.dispatch({type: "INCREMENT"})
store.dispatch({type: "DECREMENT"})
store.dispatch({type: "INCREMENT"})
store.dispatch({type: "INCREMENT"})
store.dispatch({type: "DECREMENT"})
store.dispatch({type: "INCREMENT"})
store.dispatch({type: "DECREMENT"})
