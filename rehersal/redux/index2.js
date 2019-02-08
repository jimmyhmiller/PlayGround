const createStore = (reducer, initialState) => {
    let state = reducer(initialState, {});
    let listeners = []

    return {
        getState: () => state,
        subscribe: (f) => {
            listeners.push(f);
            return () => {
                const position = listeners.indexOf(f);
                listeners.splice(position, 1)
                return undefined
            }
        },
        dispatch: (action) => {
            state = reducer(state, action)
            listeners.forEach(l => l())
            return action
        }
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
            return state
        }
    }
}

const store = createStore(reducer)
const unsub = store.subscribe(() => console.log(store.getState()))
store.dispatch({type: "INCREMENT"})
store.dispatch({type: "INCREMENT"})
store.dispatch({type: "INCREMENT"})
store.dispatch({type: "INCREMENT"})
unsub()
store.dispatch({type: "INCREMENT"})
store.dispatch({type: "INCREMENT"})
store.dispatch({type: "DECREMENT"})
