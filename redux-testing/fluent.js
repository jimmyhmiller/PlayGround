"use strict";

const zaphod = require('zaphod/compat');
const { update, set, inc, updateIn, getIn } = zaphod;
const { mapValues } = require('lodash');
const lodashColl = require('lodash/fp/collection');



var unwrapped = Symbol('unwrapped');

const identity = x => x

const fluentCompose = (combinators, f=identity) => {
  const wrapperFunction = (g) => {
    if (typeof(g) !== 'function') {
      return g;
    }
    const innerFunc = (...args) => g(...args);

    Object.keys(combinators).forEach(k => {
      const unWrappedFunc = combinators[k][unwrapped] || combinators[k];
      innerFunc[k] = (...args) => wrapperFunction(unWrappedFunc(innerFunc)(...args))
      innerFunc[k][unwrapped] = unWrappedFunc;
    })
    return innerFunc;
  }
  return wrapperFunction(f);
}

const add = getSum => (x) => () => {
  const sum = getSum() || 0;
  return sum + x;
}

const id = fluentCompose({
  add,
})

console.log(
  id
    .add(2)
    .add(3)
    .add(7)
    ()
)



let State = statefn => ({
    map: (f) => {
        return State(x => {
            let [value, state] = statefn(x);
            return [f(value), state]
        });
    },
    run: state => {
        let [value, newState] = statefn(state);
        return value;
    },
    flatMap: (f) => {
        return State(state => {
            let [value, newState] = statefn(state);
            let newStateFn = f(value).fn;
            return newStateFn(newState);
        });
    },
    then: (k) => State(statefn).flatMap(x => k),
    info: state => statefn(state),
    fn: statefn,
    get: State.get,
    put: State.put,
    pure: State.pure,
})

State.pure = (x) => {
    return State(state => {
        return [x, state];
    });
}
State.get = () => State(state => [state, state]);
State.put = (x) => State(state => [undefined, x]);


let fresh = State(n => [n,n+1]);

let freshN = State.get()
    .flatMap(n => State.put(n+1).then(State.pure(n)))


const mapState = prev => (f) => state => {
  // console.log('map', prev(state))
  const [value, newState] = prev(state);
  return [f(value), newState];
}

const flatMap = prev => (f) => state => {
  let [value, newState] = prev(state);
  let newStateFn = f(value);
  return newStateFn(newState);
}

const pure = prev => x => state => [x,  state]
const get = prev => () => state => [state, state]
const put = prev => (x) => state => [undefined, x]
const then = prev => (k) => prev.flatMap(x => k)

const runState = prev => x => {
  let [value, newState] = prev(x)
  return value;
}

const fresher = (prev) => () =>
  prev.flatMap(_ => 
    St.get()
      .flatMap(n => St.put(n+1).then(St.pure(n))))


const St = fluentCompose({
  pure,
  get: get,
  run: runState,
  map: mapState,
  flatMap,
  put,
  then,
  fresh: fresher
}, f => x => f(x))


// console.log(
// St
//   // .fresh()
//   .pure(1)
//   .fresh()
//   .fresh()
//   .fresh()
//   .fresh()
//   .run(1)
// ) // 4

const threadFirst = f => prev => (...args) => 
  coll => f(prev(coll), ...args);


const threadLast = f => prev => (...args) => 
  coll => f(...args, prev(coll))


const threadFirstAll = (obj) => mapValues(obj, threadFirst);
const threadLastAll = (obj) => mapValues(obj, threadLast);

const makefluent = (fluentModifier, f) => obj =>
  fluentCompose(fluentModifier(obj))


const fluentFirst = makefluent(threadFirstAll)
const fluentLast = makefluent(threadLastAll);

const transform = fluentFirst(zaphod)
const _ = fluentLast(lodashColl)


const fullTransform = fluentCompose({
  ..._,
  ...transform,
})



// console.log(
//   fullTransform
//     .filter(x => x % 2 === 0)
//     .map(x => x + 2)
//     .set(0, 2)
//     ([3, 1])) // [2]

const map = prev => 
  (f) => coll => {
    const newValue = prev(coll);
    if (newValue === undefined || newValue === null) {
      return newValue
    }
    return f(newValue)
}

const Maybe = fluentCompose({
  map
})

const nullable = Maybe
  .map(x => x + 2)
  .map(x => x * 3)



// console.log(nullable(0)) // 6
// console.log(nullable(null)) // null

const workflow =
  _.map(x => x + 1)
   .filter(x => x % 2 === 0)


// console.log(workflow([1,2,3,4,5])) // [2, 4, 6]

const transformer =
  transform
    .set('x', 2)
    .set('y', 3)
    .set('q', {})
    .setIn(['q', 'a'], 3)
    .updateIn(['q', 'a'], x => x + 1)

// console.log(transformer({}))
// { x: 2, y: 3, q: { a: 4 } }

// console.log(update({settings: {}}, 'settings', transformer))
//{ settings: { x: 2, y: 3, q: { a: 4 } } }



const increment = () => ({
  type: 'INCREMENT'
})

const decrement = () => ({
  type: 'DECREMENT'
})


const baseReducer = (state, action) => state;

const initialState = prev => init => (state, action) => prev(state || init, action);

const reduce = prev => (type, f) => (state, action) => {
  if (action && action.type === type) {
    return f(state, action)
  }
  return prev(state, action)
}

const run = prev => (state, action) => {
  if (action === undefined) {
    action = state;
    state = prev();
  }
  return prev(state, action)
}

const reducer = fluentCompose({ initialState, reduce, run }, baseReducer)

// console.log(
//   reducer
//   .initialState(0)
//   .reduce('INCREMENT', x => x + 1)
//   .reduce('DECREMENT', x => x - 1)
//   (0, increment())
// )


var obj = {a: {b: {c: {d: 3}}}}


// console.log(obj)








console.log('\ndone\n')