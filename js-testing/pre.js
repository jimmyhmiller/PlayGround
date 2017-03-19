

const post = x => x

const pre = (preCon, f, postCon) => (...args) => {
  if (!preCon(...args)) {
    console.error('precondition not met!');
  }
  const result = f(...args);
  if (!postCon(result)) {
     console.error('postcondition not met!');
  }
  return  result
}

const log = (...args) => {
    console.log(...args);
    return args[args.length - 1];
}

const add2 = 
pre(x => x % 2 == 0,
  x => x+2,
post(x => x % 2 == 0))

log(add2(3))