const { isZero, dec, prepend, emptyList } = require('./list.js')


const magicEq = (pred) => {
	if (typeof pred === "function") {
		return pred(true, false)
	}
	return pred;
}

const toNum = (n) => {
	if (magicEq(isZero(n))) {
		return 0;
	} else {
		return 1 + toNum(dec(n))
	}
}

const fromNum = (n) => {
	if (n === 0) {
		return zero;
	} else {
		return prepend(emptyList(), fromNum(n - 1))
	}
}

module.exports = {
	toNum,
	fromNum,
}