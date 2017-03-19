"use strict";

function Promise(fn) {
    let result = null;

    let resolve = function (value) {
        result = value;
    }
    fn(resolve)
    return {
        result: result,
        then: function(fn2) {
            let newResult = fn2(result);
            return new Promise(function (resolve) {
                resolve(newResult);
            });
        }
    }
}


function Http(url) {
    return {
        end: function(cb) {
            console.log(cb())
        }
    }
}




let y = new Promise(function (resolve) {
    new Http("someurl")
        .end(() => resolve("http!!"));
}).then(function (data) {
    console.log(data);
    console.log('got here');
})


// let x = new Promise(function (stuff) {
//     stuff(2);
// }).then(function (result) {
//     return result + 5;
// }).then(function (newResult) {
//     console.log(newResult);
//     return "hello";
// }).then(function (hello) {
//     console.log(hello);
//     return hello;
// })
