"use strict";

const zaphod = require('zaphod/compat');
const { update } = zaphod;
const lodashColl = require('lodash/fp/collection');
const { fluentCompose, threadFirst, threadLast, identity } = require('./re-fluent');








const threadFirstPrime = f => g => (...args) => x => f(g(x), ...args);


const wadd = (getSum) => x => i => getSum(i) + x
const compose = f => g => f(g());

const add = compose((sum=0) => n => () => sum + n)
const addPrime = (sum, n) => console.log('test', sum, n) || sum + n;


const adder = fluentCompose({ wadd, add, addPrime: threadFirstPrime(addPrime)  })

console.log(adder.add(2).add(3).add(3)())


const addSomeStuff = adder(adder)
  .add(3)
  .add(2)
  .addPrime(3)


const coFuncPrime = (f) => {
  const innerFunc = (x) => f(x);
}

const composePrime = (f, g) => (...args) => f(g(...args))

const coFunc = (name, f=identity) => {
  const innerFunc = (...args) => f(...args);
  innerFunc[name] = g => coFunc(name, composePrime(g, f))
  return innerFunc;
}

console.log(
  coFunc('map')
    .map(x => x + 2)
    .map(x => x + 3)
    (0)
)

// wrap :: (a -> b) -> ((a -> b) -> b -> a -> b) -> a -> b
const wrap = f => comb => x => comb(f)(x)


// (Int -> Int -> Int) -> (a -> b) -> a -> b
const c = f => g => x => f(g(x))

// const wadd = (getSum) => x => i => getSum(i) + x;
const cadd = c((sum) => x => sum + x)

// console.log(
//   wrap(identity)(cadd)(2)(0)
// )

const or = (even) => x => i => even(i) || x

// (Int -> Bool) -> Bool -> Int -> Bool
wrap((i) => i % 2 === 0)(or)(false)(2)

// console.log(
//   wrap(wrap(identity)(wadd)(3))(wadd)(1)(4)
// )


// console.log(
//   addSomeStuff(3)
// )

















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


const mapState = next => (f) => state => {
  // console.log('map', next(state))
  const [value, newState] = next(state);
  return [f(value), newState];
}

const flatMap = next => (f) => state => {
  let [value, newState] = next(state);
  let newStateFn = f(value);
  return newStateFn(newState);
}

const pure = next => x => state => [x,  state]
const get = next => () => state => [state, state]
const put = next => (x) => state => [undefined, x]
const then = next => (k) => next.flatMap(x => k)

const runState = next => x => {
  let [value, newState] = next(x)
  return value;
}

const fresher = (state) => () =>
  state.flatMap(_ => 
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


console.log(
St
  // .fresh()
  .pure(1)
  .fresh()
  .fresh()
  .fresh()
  .fresh()
  .run(1)
) // 4



const value = next => coll => next(coll);
const withValue = next => init => coll => next(init)


const transform = threadFirst(zaphod)
const _ = threadLast(lodashColl)


const fullTransform = fluentCompose({
  ..._,
  ...transform,
})

console.log('full',
  fullTransform.set(0, 2).set(1, 3).map(x => x + 2)([])
)


const randomArray = (length, max) => [...new Array(length)]
    .map(() => Math.round(Math.random() * max));

const coll = randomArray(1000, 1000);

const transformNums = fullTransform
    .filter(x => x % 2 === 0)
    .map(x => x + 2)
    .map(x => x + 2)
    .set(0, 2)

const doTransformFluent = (coll) => () => transformNums(coll)



const doTransform = (coll) => () =>
  zaphod.set(lodashColl.map(x => x + 2, lodashColl.map(x => x + 2, lodashColl.filter(x => x % 2 === 0, coll))), 0, 2)

const timeFunction = (f, name) => trials => {
  console.time(name)
  for (let i = 0; i < trials; i++) {
    f();
  }
  console.timeEnd(name);
}

timeFunction(doTransform(coll), 'not fluent')(100000)

timeFunction(doTransformFluent(coll), 'fluent')(100000)


const map = next => 
  (f) => coll => {
    if (coll === undefined || coll === null) {
      return coll
    }
    return f(next(coll))
}


const Maybe = fluentCompose({
  map
})

const nullable = Maybe
  .map(x => x + 2)
  .map(x => x * 3)


// console.log(nullable(0)) // 6
// console.log(nullable(null)) // null

// const workflow =
//   _.map(x => x + 1)
//    .filter(x => x % 2 === 0)


// console.log(workflow([1,2,3,4,5])) // [2, 4, 6]}

// console.log(update({settings: {}}, 'settings', transformer))
// //{ settings: { x: 2, y: 3, q: { a: 4 } } }



const increment = () => ({
  type: 'INCREMENT'
})

const decrement = () => ({
  type: 'DECREMENT'
})


const baseReducer = (state, action) => state;

const initialState = next => init => (state, action) => next(state || init, action);

const reduce = next => (type, f) => (state, action) => {
  if (action && action.type === type) {
    return f(state, action)
  }
  return next(state, action)
}

const run = next => (state, action) => {
  if (action === undefined) {
    action = state;
    state = next();
  }
  return next(state, action)
}

const reducer = fluentCompose({ initialState, reduce, run }, baseReducer)

console.log(
  reducer
  .initialState(0)
  .reduce('INCREMENT', x => x + 1)
  .reduce('DECREMENT', x => x - 1)
  (0, increment())
)



console.log('\ndone\n')