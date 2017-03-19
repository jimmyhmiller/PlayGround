import {
  peek as last,
  push,
  first,
  rest,
} from 'zaphod/compat';

const log = (...args) => {
  console.log(...args);
  return last(args);
}

const rightOf = maxColumns => ({x, y, w: neighborWidth}, {w: width, ...rest}) => {
  if (x + neighborWidth + width > maxColumns) {
    return {x: 0, y: y+1, w: width, ...rest}
  }
  return {x: x + neighborWidth, y, w: width, ...rest}
}

const lastReducer = (f) => (acc, x) => push(acc, f(last(acc), x))

const generateLayout = (comps, maxColumns) => {
  const firstComp = {
    x: 0,
    y: 0,
    ...first(comps),
  }
  return rest(comps).reduce(lastReducer(rightOf(maxColumns)), [firstComp])
}

const examples = [{w: 1}, {w: 1}, {w: 1}, {w: 1}, {w: 1}, {w: 1}, {w: 2}, {w: 1}, {w: 1}, {w: 1}, {w: 1}]

console.log(generateLayout(examples, 1))

console.log("reload\n\n")