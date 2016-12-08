const { Map } = require('immutable');

  ///////////////////////////////////////////
 ///               Helpers               ///
///////////////////////////////////////////

const mergeState = (mergefn) => (f) => (state, action) => mergefn(state, f(state,action))
const merge = mergeState((obj1, obj2) => Object.assign({}, obj1, obj2))
const runReducer = (reducer, ...actions) => actions.reduce(reducer, reducer())


  ///////////////////////////////////////////
 ///         Reducer Builder             ///
///////////////////////////////////////////

const identity = x => x

const getActionType = (action) => {
    if (typeof action === 'function') {
        return action.type;
    }
    return action;
}

const reducer = (reducerFns, initialState) => {
    reducerFns = reducerFns || Map({});
    const dispatcher = (state=initialState, action = {}) => {
        const f = reducerFns.get(action.type, identity);
        return f(state, action);
    }
    dispatcher.reduce = (action, fn) => reducer(reducerFns.set(getActionType(action), fn), initialState)
    dispatcher.initialState = (init) => reducer(reducerFns, init);
    return dispatcher;
}


  ///////////////////////////////////////////
 ///         Action Creator              ///
///////////////////////////////////////////

const action = (type, f=identity) => {
    const actionCreator = (obj) => Object.assign({}, {
        type: type
    }, f(obj))
    actionCreator.type = type;
    return actionCreator;
}


  ///////////////////////////////////////////
 ///               Actions               ///
///////////////////////////////////////////

const increment = action('increment');
const decrement = action('decrement');
const plus = action('add', (n) => ({ n }))
const reset = action('reset');


  ///////////////////////////////////////////
 ///     Reducing Functions              ///
///////////////////////////////////////////

const inc = ({ counter }) => ({ counter: counter + 1 })
const dec = ({ counter }) => ({ counter: counter - 1 })
const add = ({ counter }, { n }) => ({ counter: counter + n })
const res = () => ({counter: 0, hasReset: true})


  ///////////////////////////////////////////
 ///              Reducer                ///
///////////////////////////////////////////

const counterReducer = reducer()
    .initialState({ counter: 0, hasReset: false })
    .reduce(increment, merge(inc))
    .reduce(decrement, merge(dec))
    .reduce(plus, merge(add))
    .reduce(reset, res)


  ///////////////////////////////////////////
 ///              Example                ///
///////////////////////////////////////////

console.log(
    runReducer(counterReducer, 
        increment(),
        increment(),
        increment(),
        decrement(),
        reset(),
        plus(6)
    )
)

// { counter: 6, hasReset: true }

