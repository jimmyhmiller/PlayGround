const Y = f => (x => x(x))(x => f((...args) => x(x)(...args)))

const TRUE = (t, f) => t
const FALSE = (t, f) => f

const IF = (pred, t, f) => pred(t, f)

const isTrue = (boolean) => {
    return IF(boolean, TRUE, FALSE)
}


const not = (boolean) => {
    return IF(boolean, FALSE, TRUE)
}

const nothing = (x) => x

const emptyList = () => (selector) => {
    return selector(nothing, nothing, TRUE)
}

const prepend = (elem, list) => {
    return (selector) => selector(elem, list, FALSE)
}


const head = (list) => list((elem, rest) => elem)

const tail = (list) => list((elem, rest) => rest)

const isEmpty = (list) => {
    return list((elem, rest, empty) => empty)
}

const isZero = (n) => isEmpty(n)

const zero = emptyList();
const one = prepend(emptyList(), zero);
const two = prepend(emptyList(), one);

const inc = (n) => prepend(emptyList(), n)
const dec = (n) => tail(n);


const toNum = Y(f => (n) => {
    return IF(isZero(n),
        () => 0,
        () => 1 + f(dec(n))
    )()
})

const fromNum = (n) => {
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

const last = Y(f => (list) => {
    return IF(isEmpty(tail(list)),
        () => head(list),
        () => f(tail(list))
    )()
})

const countDown = Y(f => (n) => {
    return IF(isZero(n),
        () => prepend(zero, emptyList()),
        () => prepend(n, f(dec(n)))
    )()
})

const length = Y(f => (list) => {
    return IF(isEmpty(list),
        () => zero,
        () => inc(f(tail(list)))
    )()
})

// functions
// recursion


module.exports = {
    emptyList,
    prepend,
    head,
    tail,
    isEmpty,
    isTrue,
    not,
    one,
    two,
    isZero,
    toNum,
    fromNum,
    add,
    sub,
    last,
    countDown,
    length,
}