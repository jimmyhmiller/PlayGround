const Y = f => (x => x(x))(x => f((...args) => x(x)(...args)))

const TRUE = (t, f) => t

const FALSE = (t, f) => f

const IF = (pred, t, f) => pred(t, f)

const isTrue = (bool) => IF(bool, TRUE, FALSE)

const not = (bool) => IF(bool, FALSE, TRUE)



const nothing = (x) => x

const emptyList = () => {
    return (selector) => {
        return selector(nothing, nothing, TRUE)
    }
}

const prepend = (elem, coll) => {
    return (selector) => selector(elem, coll, FALSE)
}

const head = (coll) => coll((elem, coll) => elem)

const tail = (coll) => coll((elem, coll) => coll)

const isEmpty = (coll) => coll((elem, coll, empty) => {
    return empty
})

const last = Y(f => (coll) => {
    return IF(isEmpty(tail(coll)),
        () => head(coll),
        () => f(tail(coll))
    )()
})
const inc = (n) => prepend(emptyList(), n)
const dec = (n) => tail(n)

const isZero = (n) => isEmpty(n)

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

const zero = emptyList()
const one = inc(zero)
const two = inc(one)

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

const length = Y(f => (coll) => {
    return IF(isEmpty(coll),
        () => zero,
        () => inc(f(tail(coll)))
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
    inc,
    one,
    two,
    dec,
    zero,
    isZero,
    toNum,
    fromNum,
    add,
    sub,
    countDown,
    length
}