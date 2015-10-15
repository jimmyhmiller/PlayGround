var _ = require("lodash");
var $q = require("q");

var angular = {
    copy: function (item) {
        if (!item) { return item; } // null, undefined values check

        var types = [ Number, String, Boolean ],
            result;

        // normalizing primitives if someone did new String('aaa'), or new Number('444');
        types.forEach(function(type) {
            if (item instanceof type) {
                result = type( item );
            }
        });

        if (typeof result == "undefined") {
            if (Object.prototype.toString.call( item ) === "[object Array]") {
                result = [];
                item.forEach(function(child, index, array) {
                    result[index] = angular.copy( child );
                });
            } else if (typeof item == "object") {
                // testing that this is DOM
                if (item.nodeType && typeof item.cloneNode == "function") {
                    var result = item.cloneNode( true );
                } else if (!item.prototype) { // check that this is a literal
                    if (item instanceof Date) {
                        result = new Date(item);
                    } else {
                        // it is an object literal
                        result = {};
                        for (var i in item) {
                            result[i] = angular.copy( item[i] );
                        }
                    }
                } else {
                    // depending what you would like here,
                    // just keep the reference, or create new object
                    if (false && item.constructor) {
                        // would not advice to do that, reason? Read below
                        result = new item.constructor();
                    } else {
                        result = item;
                    }
                }
            } else {
                result = item;
            }
        }

        return result;
    }
}


var obj = [
    {
        name: $q.when({first: $q.when("jimmy")}),
        age: 2,
        address: $q.when({
            addressType: $q.when(2)
        })
    },
    {
        name:{first: $q.when("jimmy")},
        age: 2,
        address: $q.when({
            addressType: 2
        })
    }
]


var obj2 = {
    name: $q.when({first: $q.when("jimmy")}),
    age: 2,
    address: $q.when({
        addressType: $q.when(2)
    })
}

var obj3 = {
    name: $q.when({first: $q.when("jimmy")}),
    age: 2,
    titles: $q.when([
        {
            address: $q.when({
                addressType: $q.when(2)
            })
        },
        {
            address: $q.when({
                addressType: $q.when(2)
            })
        }
    ])
}



/*
    resolveProperty is the core of this program. It's just is very simple, resolve
    the promise. It makes sure to do this recursively in the case of arrays as well.
    On thing to note is the catch statement. If this catch statement wasn't there
    a single 404 in a chain of promise resolves would cause the whole thing to fail.
    Replacing it will null isn't the nicest, but the simplest default for now.
*/
function resolveProperty(property, obj) {
    if (_.isArray(obj)) {
        return $q.all(_.map(obj, _.partial(resolveProperty, property)));
    }
    return $q.when(obj[property])
        .catch(function () {
            return null;
        })
        .then(_.partial(setProperty, obj, property))
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
    obj[property] = value;
    return obj;
}

function getProperty(obj, property) {
    if (_.isArray(obj)) {
        return _.map(obj, property);
    }
    return obj[property];
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
function resolvePropertiesNested(obj, properties) {
    if (properties.length === 0) {
        return $q.when(obj);
    } else if (properties.length === 1) {
        return resolveProperty(_.first(properties), obj)
    }
    var currentProperty = _.first(properties);
    return resolveProperty(currentProperty, obj)
        .then(function (newObj) {
            return resolvePropertiesNested(getProperty(newObj, currentProperty), _.rest(properties))
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
function resolveMultiplePropertys(obj, properties) {
    return $q.all(_.map(properties, _.partial(resolvePropertiesNested, obj)))
        .then(_.first)
}


/*
    This is the entry point. It takes an object and an array of
    json path strings. It returns back the object with the paths specified resolved.
*/
function resolve(obj, properties) {
    var newObj = angular.copy(obj);
    return resolveMultiplePropertys(newObj, _.map(properties, function (property) {
        return property.split(".");
    }));
}

resolve(obj, ["name.first", "address.addressType"]).then(function (resolved) {
  console.log(resolved);
  console.log(obj);
})

// resolve(obj2, ["name.first", "address.addressType"]).then(function (resolved) {
//   console.log(resolved);
// })

// resolve(obj3, ["name.first", "titles.address.addressType"]).then(function (resolved) {
//   console.log(resolved);
// })





