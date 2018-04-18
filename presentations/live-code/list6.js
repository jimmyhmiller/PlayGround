
const isTrue = (bool) => bool;

const not = (bool) => !bool;

const nothing = (x) => x

const emptyList = () => (selector) => {
    return selector(nothing, nothing, true)
}

const prepend = (elem, list) => {
    return (selector) => selector(elem, list, false);
}

const head = (list) => list((elem, list) => elem);

const tail = (list) => list((elem, list) => list);

const isEmpty = (list) => list((elem, list, empty) => empty);

const isZero = (n) => isEmpty(n);

const inc = (n) => prepend(emptyList(), n);
const dec = (n) => tail(n);

const zero = emptyList();
const one = prepend(emptyList(), zero);
const two = prepend(emptyList(), one);

const toNum = (n) => {
    if (isZero(n)) {
        return 0;
    } else {
        return 1 + toNum(dec(n))
    }
}

const fromNum = (n) => {
    if (n === 0) {
        return zero
    } else {
        return inc(fromNum(n - 1))
    }
}



// functions
// booleans
// properties
// null
// equality
// strings
// if
// numbers



module.exports = {
    emptyList,
    prepend,
    head,
    tail,
    isTrue,
    not,
    isEmpty,
    isZero,
    zero,
    one,
    two,
    toNum,
    fromNum
}