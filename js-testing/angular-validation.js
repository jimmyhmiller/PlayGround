"use strict";

let _ = require("lodash/fp");
let util = require('util');


function validation(name, pred) {
    return (propValue, propName, props) => {
        return {
            name: name,
            passed: pred(propValue, propName, props)
        }
    }
}


let string = validation('string', _.isString);

let number = validation('number', _.isNumber);


function joinIfNotPassed(joiner, ...vals) {
    vals = _.filter(val => !val.passed, vals)
    return _.join(joiner, _.map('name', vals));
}


function or(validation1, validation2) {
    return (propValue, propName, props) => {
        let val1 = validation1(propValue, propName, props);
        let val2 = validation2(propValue, propName, props);
        return {
            passed: val1.passed || val2.passed,
            name: joinIfNotPassed(" || ", val1, val2)
        }
    }
}

function and(validation1, validation2) {
    return (propValue, propName, props) => {
        let val1 = validation1(propValue, propName, props);
        let val2 = validation2(propValue, propName, props);
        return {
            passed: val1.passed && val2.passed,
            name: joinIfNotPassed(" && ", val1, val2)
        }
    }
}


function gt(x) {
    return validation(`greaterThan ${x}`, (propValue) => propValue > x)
}

function dependsOn(prop) {
    return validation(`exist only if ${prop} is`, 
        (propValue, propName, props) => props[prop] !== undefined)
}

let bindings = {
    name: {
        bind: "<",
        type: string,
        required: true
    },
    age: {
        bind: "<",
        type: validation('greater than 18', x => x >= 18),
        required: false
    },
    birthday: {
        bind: "<",
        type: and(string, dependsOn('age')),
        required: false
    }
}

let values = {
    name: 'jimmy',
    age: 17,
    birthday: 23
}

let isNotRequiredAndEmpty = (propDefinition, props, propName) => {
    return propDefinition[propName].required === false && props[propName] === undefined;
}

let getValidation = (propDefinition, props, propName) => {
    return {
        key: propName,
        value: props[propName],
        validation: propDefinition[propName].type(props[propName], propName, props)
    }
}

let log = (x) => {console.log(x); return x};
let warn = (x) => {console.warn(x); return x};
let failed = _.property('validation.passed');
let not = (f) => (...x) => !f(...x);


function createMessage({value, key, validation: {name}}) {
    return `Expected property ${key} with value ${util.inspect(value, {showHidden: false, depth: null})} to be ${name}`
}

function validate(bindings, values) {
    return _.flow(
        _.keys,
        _.filter(not(_.curry(isNotRequiredAndEmpty)(bindings, values))),
        _.map(_.curry(getValidation)(bindings, values)),
        _.filter(not(failed))
    )(bindings);
}


_.flow(
    _.map(createMessage),
    _.map(warn))(validate(bindings, values))



console.log("\n\n");
