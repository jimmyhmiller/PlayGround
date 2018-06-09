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


const createStore = (reducer) => {
    let state = reducer(undefined, {});
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


const store = createStore(counter)

const unsub = store.subscribe(() => console.log(store.getState()))
store.dispatch({type: 'INCREMENT'})
store.dispatch({type: 'DECREMENT'})
store.dispatch({type: 'INCREMENT'})
store.dispatch({type: 'INCREMENT'})
store.dispatch({type: 'INCREMENT'})
store.dispatch({type: 'INCREMENT'})
unsub()
store.dispatch({type: 'INCREMENT'})
console.log('\n\n')