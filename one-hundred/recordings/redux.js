

const applyMiddleware = (middleware) => (createStore) => (...args) => {
    const store = createStore(...args)
    const dispatch = middleware(store)(store.dispatch);
    return {
        ...store,
        dispatch
    }
}

const loggingMiddleware = ({ getState }) => (next) => action => {
    console.log("Before:", getState())
    const result = next(action);
    console.log("After:", getState());
    return result
}

const createStore = (reducer, initialState, enhancer) => {
    if (enhancer) {
        return enhancer(createStore)(reducer, initialState)
    }

    let state = reducer(initialState, {});
    let listeners = []
    return {
        getState: () => state,
        dispatch: (action) => {
            state = reducer(state, action);
            listeners.forEach(f => f())
            return action;
        },
        subscribe: (listener) => {
            listeners.push(listener);
            const fsPosition = listeners.length - 1;
            return () => listeners.splice(fsPosition, 1)
        }
    }
}




////////////////////////////////////////////////////
const increment = () => ({
    type: 'INCREMENT'
})
const decrement = () => ({
    type: 'DECREMENT'
})

const reducer = (state=0, action) => {
    switch (action.type) {
        case 'INCREMENT': {
            return state + 1
        }
        case 'DECREMENT': {
            return state - 1;
        }
        default: {
            return state
        }
    }
}

const store = createStore(reducer, 10)

const unsub = store.subscribe(() => console.log(store.getState()))
store.dispatch(increment())
store.dispatch(increment())
store.dispatch(increment())
store.dispatch(increment())
store.dispatch(increment())
unsub()
store.dispatch(decrement())
store.dispatch(decrement())
store.dispatch(decrement())





