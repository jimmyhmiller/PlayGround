

const TRUE = (t, f) => t;
const FALSE = (t, f) => f;
const IF = (pred, t, f) => pred(t, f);

const Y = f => (x => x(x))(x => f((...args) => x(x)(...args)))

const isTrue = (b) => IF(b, TRUE, FALSE);
const not = (b) => IF(b, FALSE, TRUE);
const nothing = (x) => x;

const emptyList = () => (selector) => 
	selector(nothing, nothing, TRUE)

const prepend = (elem, list) =>
	(selector) => selector(elem, list, FALSE);

const head = (list) => list((elem, list) => elem);

const tail = (list) => list((elem, list) => list);

const isEmpty = (list) => 
	list((elem, list, empty) => empty);

const last = Y(f => (list) => {
	return IF(isEmpty(tail(list)),
		() => head(list),
		() => f(tail(list))
	)()
})


const isZero = (n) => isEmpty(n);
const inc = (n) => prepend(emptyList(), n);
const dec = (n) => tail(n);

const toNum = Y(f => (n) => {
	return IF(isEmpty(n),
		() => 0,
		() => 1 + f(dec(n))
	)()
})

const zero = emptyList();
const one = inc(zero);
const two = inc(one);

function fromNum(n) {
	if (n === 0) {
		return zero
	} else {
		return inc(fromNum(n - 1))
	}
}

const add = Y(f => (a, b) => {
	return IF(isZero(a),
		() => b,
		() => f(dec(a), inc(b))
	)()
})

const sub = Y(f => (a, b) => {
	return IF(isZero(b),
		() => a,
		() => f(dec(a), dec(b))
	)()
})

const countDown = Y(f => (n) => {
	return IF(isZero(n),
		() => prepend(zero, emptyList()),
		() => prepend(n, f(dec(n)))
	)()
})

const nth = Y(f => (list, n) => {
	return IF(isZero(n),
		() => head(list),
		() => f(tail(list), dec(n))
	)()
})

const length = Y(f => (list) => {
	return IF(isEmpty(list),
		() => zero,
		() => inc(f(tail(list)))
	)()
})

// functions


module.exports = {
	emptyList,
	prepend,
	head,
	tail,
	isEmpty,
	isTrue,
	not,
	last,
	isZero,
	zero,
	one,
	two,
	inc,
	dec,
	toNum,
	fromNum,
	add,
	sub,
	countDown,
	nth,
	length
}