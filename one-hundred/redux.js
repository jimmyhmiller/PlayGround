console.log('\033c')


const compose = (f, g) => (x) => f(g(x))

const applyMiddleware = (...middlewares) => (createStore) => (...args) => {
    const store = createStore(...args)
    const dispatch = middlewares
        .map(m => m(store))
        .reduce(compose)(store.dispatch)
    return Object.assign(
        {},
        store,
        {dispatch: dispatch}
    )
}


const createStore = (reducer, initialState, enhancer) => {
    if (enhancer) {
        return enhancer(createStore)(reducer, initialState)
    }
    let state = reducer(initialState, {});
    let listeners = [];
    return {
        subscribe: (f) => { 
            listeners.push(f);
            const fsPosition = listeners.length - 1 
            return () => listeners.splice(fsPosition, 1);
        },
        getState: () => state,
        dispatch: (action) => {
            state = reducer(state, action)
            listeners.forEach(l => l());
            return action
        },
    }
}

/////////////////////////////////////////////////

const counter = (state=0, action) => {
    switch (action.type) {
        case 'INCREMENT': {
            return state + 1;
        }
        case 'DECREMENT': {
            return state - 1;
        }
        default: {
            return state
        }
    }
}

function logger({ getState }) {
  return next => action => {
    console.log('will dispatch', action)

    const returnValue = next(action)
    console.log('state after dispatch', getState())

    return returnValue
  }
}


const store = createStore(counter, 12, applyMiddleware(logger))

const unsub = store.subscribe(() => console.log(store.getState()))
store.dispatch({type: 'INCREMENT'})
store.dispatch({type: 'DECREMENT'})
store.dispatch({type: 'INCREMENT'})
store.dispatch({type: 'INCREMENT'})
store.dispatch({type: 'INCREMENT'})
store.dispatch({type: 'INCREMENT'})
unsub()
store.dispatch({type: 'INCREMENT'})
