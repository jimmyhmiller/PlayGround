

const TRUE = (t, f) => t;
const FALSE = (t, f) => f;
const IF = (pred, t, f) => pred(t, f); 

const Y = f => (x => x(x))(x => f((...args) => x(x)(...args)))

const nothing = (x) => x;

const emptyList = () => (command) => 
  command(nothing, nothing, TRUE)

const prepend = (elem, list) => 
  (command) => command(elem, list, FALSE)

const head = (list) => list((elem, list) => elem);

const tail = (list) => list((elem, list) => list);

const isEmpty = (list) => 
  list((elem, list, empty) => empty)

const not = (x) => IF(x, FALSE, TRUE)

const isTrue = (x) => IF(x, TRUE, FALSE);

const toNum = Y(f => (n) =>
	IF (isZero(n),
		() => 0,
		() => 1 + f(dec(n)))())


const fromNum = (n) => {
	if (n === 0) {
		return zero;
	} else {
		return prepend(emptyList(), fromNum(n - 1))
	}
}

const isZero = isEmpty;
const zero = emptyList();
const one = prepend(emptyList(), zero);
const two = prepend(emptyList(), one);

const inc = (n) => prepend(emptyList(), n);
const dec = (n) => tail(n);

const add = Y(f => (a, b) =>
	IF (isZero(a),
		() => b,
		() => f(dec(a), inc(b)))())

const sub = Y(f => (a, b) => 
	IF (isZero(b),
		() => a,
		() => f(dec(a), dec(b)))())

const mult = Y(f => (a, b) => 
	IF (isZero(b),
		() => zero,
		() => add(a, f(a, dec(b))))())


const countDown = Y(f => (n) =>
	IF (isZero(n),
		() => prepend(zero, emptyList()),
		() => prepend(n, f(dec(n))))())

const nth = Y(f => (list, n) =>
	IF (isZero(n),
		() => head(list),
		() => f(tail(list), dec(n)))())

const length = Y(f => (list) =>
	IF (isEmpty(list),
		() => zero,
		() => inc(f(tail(list))))())

module.exports = {
  prepend,
  head,
  tail,
  emptyList,
  isEmpty,
  zero,
  one,
  inc,
  dec,
  toNum,
  fromNum,
  add,
  sub,
  mult,
  nth,
  countDown,
  length,
  not,
  isTrue,
}

// functions