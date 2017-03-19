"use strict";


function pprint(obj) {
    console.log(JSON.stringify(obj, null, 4))
    return obj;
}

let fetch = require("node-fetch");




fetch("https://www.reddit.com/.jso")
    .then(function (data) {
        if (data.status >= 300) {
            return Promise.reject(data.status);
        }
        return data;
    })
    .then(function (data) {
        console.log("success!");
        return data;
    })
    .catch(function (err) {
        console.log(err);
        return Promise.reject(err);
    })
    .then(function (data) {
        console.log(data);
        return data;
    })















console.log("");