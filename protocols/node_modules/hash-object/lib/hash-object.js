/*
 *
 * https://github.com/davidcaseria/hash-object
 *
 * Copyright (c) 2014 David Caseria
 * Licensed under the MIT license.
 */

'use strict';

var crypto = require('crypto');

function getType(obj) {
    if (typeof obj === 'object') {
        if (Array.isArray(obj) && obj.length > 0) {
            return 'array';
        } else {
            for (var key in obj) {
                if (obj.hasOwnProperty(key)) {
                    return 'object';
                }
            }
            return null;
        }
    } else {
        return (typeof obj);
    }
}

function stringify(obj) {
    try {
        var keys = Object.keys(obj);
        keys.sort();

        var str = '';
        for (var i = 0; i < keys.length; i++) {
            var key = keys[i];
            var type = getType(obj[key]);

            if (type === 'object') {
                str += '|' + key + ':' + stringify(obj[key]) + '|';
            } else if (type === 'array') {
                var elements = [];
                for (var j = 0; j < obj[key].length; j++) {
                    var s = stringify(obj[key][j]);
                    if (s) {
                        elements.push(s);
                    }
                }
                if (elements.length > 0) {
                    str += key + ':[' + elements.join(',') + ']';
                }
            } else if (type === 'string' || type === 'number' || type === 'boolean') {
                str += '|' + key + ':' + obj[key] + '|';
            }
        }

        return str;
    } catch (error) {
        return null;
    }
}

module.exports = function (obj, opts) {
    opts = typeof opts === 'undefined' ? {} : opts;

    if (typeof obj !== 'object' || !obj) {
        return null;
    }

    try {
        var hash = crypto.createHash(opts.algorithm ? opts.algorithm : 'sha1');
        //console.log(stringify(obj));
        hash.update(stringify(obj));
        return hash.digest('hex');
    } catch (err) {
        return null;
    }
};