/*
 * This is a JavaScript Scratchpad.
 *
 * Enter some JavaScript, then Right Click or choose from the Execute Menu:
 * 1. Run to evaluate the selected text (Cmd-R),
 * 2. Inspect to bring up an Object Inspector on the result (Cmd-I), or,
 * 3. Display to insert the result in a comment after the selection. (Cmd-L)
 */

const READ = 'READ';
const WRITE = 'WRITE';
const CALL = 'CALL';

const read = () => ({
  type: READ,
})

const write = (text) => ({
  type: WRITE,
  text,
})

const call = (f, ...args) => ({
  type: CALL,
  f,
  args,
})

let interpreter = (action) => {
  switch (action.type) {
    case READ:
      const text = window.prompt()
      return Promise.resolve(text)
    case WRITE:
      console.log(action.text);
      return undefined;
    case CALL:
      return action.f(...action.args);
    default:
      return "NO COMMAND!!!"
  }
}

const called = function* () {
  const called = yield call(x => x, 'called!');
  yield write(called);
}


const echo = function* () {
  const text = yield read();
  yield write(text);
}


const resolveValue = (value) => Promise.resolve(value)


async function run(interpreter, command) {
  const gen = command();
  let next = gen.next();
  while (true) {
    if (next.value) {
      let newValue = await resolveValue(interpreter(next.value));
      if (newValue) {
        next = gen.next(newValue);
      }
      else {
        next = gen.next()
      }
    }
    if (next.done) {
      break;
    }
  }
}


run(interpreter, echo)
