import { createStore, bindActionCreators } from 'redux'
import { useState, useEffect, useMemo } from 'react';

// from the docs
function todos(state = [], action) {
  switch (action.type) {
    case 'ADD_TODO':
      return state.concat([{text: action.text}])
    default:
      return state
  }
}

export const addTodo = (text) => ({
  type: 'ADD_TODO',
  text
})

const createReduxHook = (reducer, initialState, enhancer) => {
    const store = createStore(reducer, initialState, enhancer);
    return {
        useActions: (actionCreators) => bindActionCreators(actionCreators, store.dispatch),
        useSelector: (selector) => {
            const selectMemo = useMemo(selector, [store])
            const [value, setValue] = useState(null);
            useEffect(() => {
                setValue(selector(store.getState()))
                return store.subscribe(() => {
                    setValue(selector(store.getState()));
                })
            }, [selector.toString()])
            return value;
        }
    }
}
export const { useSelector, useActions } = createReduxHook(todos, [{text: 'UseRedux'}]);

