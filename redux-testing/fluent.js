const zaphod = require('zaphod/compat');
const { update } = zaphod;
const { mapValues } = require('lodash');
const lodashColl= require('lodash/fp/collection');

const fluentCompose = (f, combinators) => {
  const wrapperFunction = (g) => {
    if (typeof(g) !== 'function') {
      return g;
    }
    const innerFunc = (...args) => g(...args);
    Object.keys(combinators).forEach(k => {
      innerFunc[k] = (...args) => wrapperFunction(combinators[k](g)(...args))
    })
    return innerFunc;
  }
  return wrapperFunction(f);
}

const threadFirst = f => next => 
  (...args) => coll => f(next(coll), ...args);

const threadLast = f => next =>
  (...args) => coll => f(...args, next(coll));


const threadFirstAll = (obj) => mapValues(obj, threadFirst);
const threadLastAll = (obj) => mapValues(obj, threadLast);


const identity = coll => coll
const value = next => coll => next(coll);
const withValue = next => init => coll => next(init)

const makefluent = (fluentModifier, f=identity) => obj =>
  fluentCompose(f, {
    ...fluentModifier(obj),
    value,
    withValue,
  })


const fluentFirst = makefluent(threadFirstAll)
const fluentLast = makefluent(threadLastAll);

const transform = fluentFirst(zaphod)
const _ = fluentLast(lodashColl)


const workflow =
  _.map(x => x + 2)
   .filter(x => x % 2 === 0)
   .reduce((a, b) => a + b, 0)


console.log(workflow([1,2,3,4,5]))

const transformer =
  transform
    .set('x', 2)
    .set('y', 3)
    .set('q', {})
    .setIn(['q', 'a'], 3)
    .updateIn(['q', 'a'], x => x + 1)

console.log(transformer({}))
// { x: 2, y: 3, q: { a: 4 } }

update({settings: {}}, 'settings', transformer)
//{ settings: { x: 2, y: 3, q: { a: 4 } } }


console.log('\ndone\n')