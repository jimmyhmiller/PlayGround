var _ = require("lodash");
var $q = require("q"); 
var fetch = require("node-fetch");
fetch.Promise = require('bluebird');



/*
    resolveProperty is the core of this program. It's just is very simple, resolve
    the promise. It makes sure to do this recursively in the case of arrays as well.
    On thing to note is the catch statement. If this catch statement wasn't there
    a single 404 in a chain of promise resolves would cause the whole thing to fail.
    Replacing it will null isn't the nicest, but the simplest default for now.
*/
function resolveProperty(headers, property, obj) {
    console.log(property);
    if (_.isArray(obj)) {
        return $q.all(_.map(obj, _.partial(resolveProperty, headers, property)));
    }
    var value = getProperty(obj, property);
    console.log(value);
    if (value.href != undefined) {
        return fetch(value.href, {headers: headers})
            .catch(function () {
                console.log('fail3');
            })
            .then(function (res) {
                if (res.status >= 300) {
                    console.log('got here');
                    return null;
                }
                return res.json();
            })
            .catch(function () {
                console.log('fail2');
            })
            .then(_.partial(setProperty, obj, property))
            .catch(function () {
                console.log('fail');
            })
    } else {
      return $q.when(value)
        .then(_.partial(setProperty, obj, property))
        .catch(function () {
            return setProperty(obj, property, null);
        });
    }
}


/*
    setProperty is fairly basic. It sets the property on the object,
    or if it is an array sets the property recursively. One thing to note,
    if obj is an array, value will be an array as well. They will have an
    index to index match. So we need to set the object to its corresponding
    value in the values array.
*/
function setProperty(obj, property, value) {
    if (_.isArray(obj) && _.isArray(value)) {
        return _.map(obj, function (o, i) {
            return setProperty(o, property, value[i]);
        })
    }
    if (value && value._embedded && value._embedded[property]) {
        obj[property] = value._embedded[property];
    } else {
        obj[property] = value;
    }
    return obj;
}

function getProperty(obj, property) {
    if (_.isArray(obj)) {
        return _.map(obj, property);
    }
    if (obj[property] != undefined) {
        return obj[property];
    } else {
        return obj._links[property];
    }
  
}

/*
    This is the powerhouse of the program. It may look a bit
    scary but isn't too difficult. The base case looks like this:
        obj = {a: Promise({})}
        properties = ['a']
    This base case is incredly simple, we call resolveProperty and 
    get back the object with it's one property resolved. The recursive
    case its a bit harder.
        obj = {a: Promise({b: Promise({})})}
        properties = ['a', 'b']
    We first do the exact same thing we resolve 'a' getting back:
        newObj = {a: {b: Promise({})}}
    Then we recursively resolve the rest of the properties. In other
    words we will call resolvePropertiesNested with:
        obj = {b: Promise({})}
        properties = ['b']
    We get back the resolved promise:
        {b: {}}
    And then finally we set our original property equal to the 
    recursively resolved object. 

*/
function resolvePropertiesNested(headers, obj, properties) {
    if (properties.length === 0) {
        return $q.when(obj);
    } else if (properties.length === 1) {
        return resolveProperty(headers, _.first(properties), obj)
    }
    var currentProperty = _.first(properties);
    return resolveProperty(headers, currentProperty, obj)
        .then(function (newObj) {
            return resolvePropertiesNested(headers, getProperty(newObj, currentProperty), _.rest(properties))
                .then(_.partial(setProperty, newObj, currentProperty))
        });
}

/* 
    The goal here is just resolve multiple sets of properties.
    For instance:
        obj = {
            a: Promise({b: Promise({})}),
            c: Promise({d: Promise({})})
        }
        properties = [['a', 'b'], ['c', 'd']]
    The _.first here may seem a bit weird.
    In fact it wasn't what I expected to work.
    The reason it does is two fold.
    1) Objects are references and so even when done in parallel 
       the changes affect it globally.
    2) $q.all makes sure that we are done with all modifications.
       So what the $q.all returns is actually the same object n times.
       Since they are the same object, we can just take the first.
 */
function resolveMultiplePropertys(headers, obj, properties) {
    return $q.all(_.map(properties, _.partial(resolvePropertiesNested, headers, obj)))
        .then(_.first) 
}


/*
    This is the entry point. It takes an object and an array of 
    json path strings. It returns back the object with the paths specified resolved.
*/
function resolve(headers, obj, properties) {
    return resolveMultiplePropertys(headers, obj, _.map(properties, function (property) {
        return property.split(".");
    }));
}


module.exports = {
    resolve: resolve
}

