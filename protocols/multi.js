"use strict";

let _ = require("lodash");
let hash = require("object-hash");

function multi(fn) {
    let m = new Map();
    let defaultMethod = function(...args) {
        throw new Error("No match found and no default");
    } 
    function dispatcher(...args) {
        let value = fn(...args);
        if (value != undefined && m.has(hash.sha1(value))) {
           return m.get(hash.sha1(value))(...args); 
        } else {
            return defaultMethod(...args);
        }
        
    }
    dispatcher.method = (value, f) => {
        m.set(hash.sha1(value), f);
        return this;
    }
    dispatcher.defaultMethod = (f) => {
        defaultMethod = f;
        return this;
    } 
    return dispatcher;
}


var circle = (radius) => ({shape: 'circle', radius});
var rect = (width, height) => ({shape: 'rect', width, height});


var area = multi(_.property('shape'));

area.method('rect', r => r.width * r.height)
area.method('circle', c => 3.14 * (c.radius * c.radius))

area.method({a: 1}, x => console.log('it worked!'));

area.defaultMethod(x => console.log('default'));

area({shape: {a: 1}})

console.log(
   area(rect(4,13))
);
console.log(
    area(circle(12))
);

// console.log('yep');