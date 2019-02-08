


const createStore = (reducer, initialState, enhancer) => {
    if (enhancer) {
        return enchancer(createStore)(reducer, initialState)
    }
    let state = reducer(initialState, {});
    let listeners = [];

    return {
        getState: () => state,
        subscribe: (f) => {
            listeners.push(f);
            return () => {
                const position = listeners.indexOf(f);
                listeners.splice(position, 1)
                return undefined;
            }
        },
        dispatch: (action) => {
            state = reducer(state, action)
            listeners.forEach(f => f())
            return action
        }
    }
}





const logger = ({ getState }) => {
  return next => action => {
    console.log('will dispatch', action)

    const newState = next(action)
    console.log('state after dispatch', getState())

    return newState
  }
}




const reducer = (state=0, action) => {
    switch (action.type) {
        case "INCREMENT": {
            return state + 1;
        }
        case "DECREMENT": {
            return state - 1;
        }
        default: {
            return state;
        }
    }
}



const store = createStore(reducer, undefined, )

const unsub = store.subscribe(() => console.log(store.getState()))
