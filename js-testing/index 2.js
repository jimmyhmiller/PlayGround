"use strict";


let Y = (f) =>
    (x => x(x))(x => 
        f(y => x(x)(y)));


let Y2 = function (f) {
    return (function (x) {
        return x(x)
    })(function (x) {
        return f(function (y) {
            return x(x)(y);
        });
    });
}


let fib = (f) => (n) => {
    if (n === 0) {
        return 0;
    } else if (n === 1) {
        return 1
    } else {
        return f(n-1) + f(n-2)
    }
}


console.log(fib(Y(fib))(5))



let fresh = n => [n, n+1];

let f = function*() {
    let x = yield(fresh);
    let y = yield(fresh);
    yield(s => [[x,y], s]);
}


function runGenState(s, gen) {
    gen = gen();
    while (true) {
        let fn = gen.next();
        if (fn.done) {
            return s[1];
        }
        s = fn.value(s);
        gen.next(s[0]); 
    }
}



let State = statefn => ({
    map: (f) => {
        return State(x => {
            let [value, state] = statefn(x);
            return [f(value), state]
        });   
    },
    run: state => {
        let [value, newState] = statefn(state);
        return value;
    },
    flatMap: (f) => {
        return State(state => {
            let [value, newState] = statefn(state);
            let newStateFn = f(value).fn;
            return newStateFn(newState);
        });
    },
    then: (k) => State(statefn).flatMap(x => k),
    info: state => statefn(state),
    fn: statefn,
    get: State.get,
    put: State.put,
    pure: State.pure,
})

State.pure = (x) => {
    return State(state => {
        return [x, state];
    });
}
State.get = () => State(state => [state, state]);
State.put = (x) => State(state => [undefined, x]);


let fresh = State(n => [n,n+1]);

let freshN = State.get()
    .flatMap(n => State.put(n+1).then(State.pure(n)))

// console.log(
//   fresh.flatMap(function (a) {
//     return fresh.flatMap(function (b) {
//         return fresh.flatMap(function (c) {
//             return State.pure([a,b,c]);
//         });
//     });
// })
//     .run(10)
// )