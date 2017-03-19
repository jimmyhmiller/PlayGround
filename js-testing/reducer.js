const multi = require('multiple-methods');



const log = (...args) => {
    console.log(...args);
    return args[args.length - 1];
}

const identity = x => x


const action = (type, f) => {
    f = f || identity;
    const actionCreator = (obj) => Object.assign({}, {
        type: type
    }, f(obj))

    actionCreator.type = type;
    return actionCreator;
}

const mergeState = (mergefn) => (f) => (state, action) => mergefn(state, f(state,action))
const merge = mergeState((obj1, obj2) => Object.assign({}, obj1, obj2))


const reducer = multi((_, action) => action && action.type);

reducer.reduce = (actionOrType, fn) => {
    if (typeof action === 'function') {
        return reducer.method(actionOrType.type, fn);
    } else {
        return reducer.method(actionOrType, fn);
    }
}


reducer.initalState = (initialState) => reducer.defaultMethod((state, action) => {
    if (!action) {
        return initialState
    } else {
        return state
    }
})









////////////////////////////////////////
///         Actions                 ///
//////////////////////////////////////

const increment = action('increment');
const decrement = action('decrement');
const plus = action('add', (n) => ({ n }))
const reset = action('reset');



////////////////////////////////////////
///         Reducer                 ///
//////////////////////////////////////

const inc = ({ counter }) => ({ counter: counter + 1 })
const dec = ({ counter }) => ({ counter: counter - 1 })
const add = ({ counter }, { n }) => ({ counter: counter + n })
const res = () => ({counter: 0, hasReset: true})

const counterReducer = reducer
    .initalState({ counter: 0, hasReset: false })
    .reduce(increment, merge(inc))
    .reduce(decrement, merge(dec))
    .reduce(plus, merge(add))
    .reduce(reset, res)



////////////////////////////////////////
///         Example                 ///
//////////////////////////////////////



const runReducer = (reducer, ...actions) => actions.reduce(reducer, reducer())


log(runReducer(counterReducer, 
    increment(),
    increment(),
    increment(),
    decrement(),
    reset(),
    plus(6)))



log('refresh\n\n\n')