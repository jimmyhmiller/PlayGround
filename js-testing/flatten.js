const { reduce }  = require('lodash');
const { set, merge } = require('zaphod/compat');

const makeKey = (prefix, key) => {
  if (prefix === "") {
    return key;
  }
  return `${prefix}_${key}`
}

const mergeRecursive = prefix => (result, value, key) => {
  const newKey = makeKey(prefix, key);
  if (typeof value !== 'object') {
    return set(result, newKey, value);
  }
  return merge(result, flatten(value, newKey))
}

const flatten = (x, prefix='') => reduce(x, mergeRecursive(prefix), {})


var x = {
  x: {
    y: 1,
    z: 3,
    a: {
      b: 2,
      c: {
        q: 7
      }
    }
  },
  z: '3'
}


console.log(flatten(x))
