
const compose = (f, g) => (...args) => (f(g(...args)))

const applyMiddleware = (...middlewares) => (createStore) => (...args) => {
    const store = createStore(...args);
    const dispatch = middlewares
        .map(m => m(store))
        .reduce(compose)(store.dispatch);
    return {
        ...store,
        dispatch
    }
}

const createStore = (reducer, initialState, enhancer) => {
    if (enhancer) {
        return enhancer(createStore)(reducer, initialState);
    }

    let state = reducer(initialState, {})
    let listeners = [];
    return {
        getState: () => state,
        dispatch: (action) => {
            state = reducer(state, action);
            listeners.forEach(f => f());
            return action
        },
        subscribe: (listener) => {
            listeners.push(listener);
            const listenerPosition = listeners.length - 1;
            return () => listeners.splice(listenerPosition, 1);
        }
    }
}




////////////////////////


// const increment = () => ({
//     type: "INCREMENT"
// })

// const decrement = () => ({
//     type: "DECREMENT"
// })

// const reducer = (state=0, action) => {
//     switch (action.type) {
//         case "INCREMENT": {
//             return state + 1;
//         }
//         case "DECREMENT": {
//             return state - 1;
//         }
//         default: {
//             return state
//         }
//     }
// }
// const enhancer = applyMiddleware(
//     loggingMiddleware
// );

// const store = createStore(reducer, 10, enhancer)
// store.dispatch(increment())
// store.dispatch(increment())
// store.dispatch(increment())
// store.dispatch(increment())
// store.dispatch(increment())
// store.dispatch(increment())
// store.dispatch(increment())
// store.dispatch(increment())
// store.dispatch(decrement())
// store.dispatch(decrement())




