const { mapValues } = require('lodash');

const identity = x => x

const fluentCompose = (combinators, f=identity) => {
  if (typeof(f) !== 'function') {
    return f;
  }

  const innerFunc = (...args) => f(...args);
  const methods = mapValues(combinators, g => 
    (...args) => fluentCompose(combinators, g(innerFunc)(...args)))

  return Object.assign(innerFunc, methods);
 }

const threadFirstSingle = f => next => (...args) => {
  return coll => f(next(coll), ...args);
}

const threadLastSingle = f => next => (...args) => {
   return coll => f(...args, next(coll))
}

const threadFirst = (obj) => mapValues(obj, threadFirstSingle);
const threadLast = (obj) => mapValues(obj, threadLastSingle);



module.exports = {
  fluentCompose,
  threadFirst,
  threadLast,
  identity,
}
