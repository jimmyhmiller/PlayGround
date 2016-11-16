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
