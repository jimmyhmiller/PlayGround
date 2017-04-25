"use strict";

var unwrapped = Symbol('unwrapped');

const identity = x => x

const fluentCompose = (combinators, f=identity) => {
  const wrapperFunction = (g) => {
    if (typeof(g) !== 'function') {
      return g;
    }
    const innerFunc=(...args) => g(...args);

    Object.keys(combinators).forEach(k => {
      const unWrappedFunc = combinators[k][unwrapped] || combinators[k];
      innerFunc[k] = (...args) => wrapperFunction(unWrappedFunc(innerFunc)(...args))
      innerFunc[k][unwrapped] = unWrappedFunc;
    })
    return innerFunc;
  }
  return wrapperFunction(f);
}


const map = f => parser => s => parser(s).map(([a, b]) => [f(a), b])
const pure = a => s => [[a, s]]


console.log('done\n')